import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

import sidechainnet as scn
from sidechainnet.dataloaders.collate import prepare_dataloaders
from alphafold2_pytorch import Alphafold2
import alphafold2_pytorch.constants as constants
from alphafold2_pytorch.utils import get_bucketed_distance_matrix
from alphafold2_pytorch.transformer import Seq2SeqTransformer
import time

# constants

DEVICE = None  # defaults to cuda if available, else cpu
NUM_EPOCHS = int(1e3)
NUM_BATCHES = int(1e5)
GRADIENT_ACCUMULATE_EVERY = 16
LEARNING_RATE = 3e-4
IGNORE_INDEX = -100
THRESHOLD_LENGTH = 50

# set device

DISTOGRAM_BUCKETS = constants.DISTOGRAM_BUCKETS
DEVICE = constants.DEVICE


# helpers


def cycle(loader, cond=lambda x: True):
    while True:
        for data in loader:
            if not cond(data):
                continue
            yield data


def filter_dictionary_by_seq_length(raw_data, seq_length_threshold, portion):
    """Filter SidechainNet data by removing poor-resolution training entries.

    Args:
        raw_data (dict): SidechainNet dictionary.
        seq_length_threshold (int): sequence length threshold

    Returns:
        Filtered dictionary.
    """
    new_data = {
        "seq": [],
        "ang": [],
        "ids": [],
        "evo": [],
        "msk": [],
        "crd": [],
        "sec": [],
        "res": []
    }
    train = raw_data[portion]
    n_filtered_entries = 0
    total_entires = 0.
    for seq, ang, crd, msk, evo, _id, res, sec in zip(train['seq'], train['ang'],
                                                      train['crd'], train['msk'],
                                                      train['evo'], train['ids'],
                                                      train['res'], train['sec']):
        total_entires += 1
        if len(seq) > seq_length_threshold:
            n_filtered_entries += 1
            continue
        else:
            new_data["seq"].append(seq)
            new_data["ang"].append(ang[:, 0:3])
            new_data["ids"].append(_id)
            new_data["evo"].append(evo)
            new_data["msk"].append(msk)
            new_data["crd"].append(crd)
            new_data["sec"].append(sec)
            new_data["res"].append(res)
    if n_filtered_entries:
        print(
            f"{total_entires - n_filtered_entries:.0f} out of {total_entires:.0f} ({(total_entires - n_filtered_entries) / total_entires:.1%})"
            f" training set entries were included if sequence length <= {seq_length_threshold}")
    raw_data[portion] = new_data
    return raw_data


def create_mask(src, tgt):
    src_padding_mask = (src == IGNORE_INDEX).transpose(0, 1)
    tgt_padding_mask = (tgt == IGNORE_INDEX).transpose(0, 1)
    return src_padding_mask, tgt_padding_mask


def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, (batch) in enumerate(train_iter):
        seq, coords, angs, mask = batch.seqs, batch.crds, batch.angs, batch.msks

        b, l, _ = seq.shape

        # prepare mask, labels

        seq, coords, angs, mask = seq.argmax(dim=-1).to(DEVICE), coords.to(DEVICE), angs.to(DEVICE), mask.to(
            DEVICE).bool()
        seq = F.pad(seq, (0, THRESHOLD_LENGTH - l), value=0)
        coords = rearrange(coords, 'b (l c) d -> b l c d', l=l)
        angs = F.pad(angs, (0, 0, 0, THRESHOLD_LENGTH - l), value=0)
        # angs = rearrange(angs, 'b l c -> b (l c)', l=THRESHOLD_LENGTH)
        mask = F.pad(mask, (0, THRESHOLD_LENGTH - l), value=False)

        # discretized_distances = get_bucketed_distance_matrix(coords[:, :, 1], mask, DISTOGRAM_BUCKETS, IGNORE_INDEX)
        src_padding_mask, tgt_padding_mask = create_mask(seq, seq)

        # predict

        logits = transformer(seq, seq, src_mask=mask,
                             tgt_mask=mask, src_padding_mask=src_padding_mask,
                             tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)

        # loss
        optimizer.zero_grad()

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), angs.reshape(-1, angs.shape[-1]))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    return losses / len(train_iter)


def evaluate(model, val_iter):
    model.eval()
    losses = 0
    for idx, (batch) in (enumerate(val_iter)):
        seq, coords, angs, mask = batch.seqs, batch.crds, batch.angs, batch.msks

        b, l, _ = seq.shape

        # prepare mask, labels

        seq, coords, angs, mask = seq.argmax(dim=-1).to(DEVICE), coords.to(DEVICE), angs.to(DEVICE), mask.to(
            DEVICE).bool()
        seq = F.pad(seq, (0, THRESHOLD_LENGTH - l), value=0)
        coords = rearrange(coords, 'b (l c) d -> b l c d', l=l)
        angs = F.pad(angs, (0, 0, 0, THRESHOLD_LENGTH - l), value=0)
        # angs = rearrange(angs, 'b l c -> b (l c)', l=THRESHOLD_LENGTH)
        mask = F.pad(mask, (0, THRESHOLD_LENGTH - l), value=False)

        # discretized_distances = get_bucketed_distance_matrix(coords[:, :, 1], mask, DISTOGRAM_BUCKETS, IGNORE_INDEX)
        src_padding_mask, tgt_padding_mask = create_mask(seq, seq)

        # predict

        logits = transformer(seq, seq, src_mask=mask,
                             tgt_mask=mask, src_padding_mask=src_padding_mask,
                             tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)

        # loss

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), angs.reshape(-1, angs.shape[-1]))
        loss.backward()

        losses += loss.item()
    return losses / len(val_iter)


# get data

raw_data = scn.load(
    casp_version=12,
    thinning=30,
    batch_size=1,
    dynamic_batching=False
)

filtered_raw_data = filter_dictionary_by_seq_length(raw_data, THRESHOLD_LENGTH, "train")
writer_train = SummaryWriter("runs/train")
writer_valids = []
for split in scn.utils.download.VALID_SPLITS:
    filtered_raw_data = filter_dictionary_by_seq_length(filtered_raw_data, THRESHOLD_LENGTH, f'{split}')
    writer_valids.append(SummaryWriter(f"runs/{split}"))
data = prepare_dataloaders(
    filtered_raw_data,
    aggregate_model_input=True,
    batch_size=1,
    num_workers=4,
    seq_as_onehot=None,
    collate_fn=None,
    dynamic_batching=False,
    optimize_for_cpu_parallelism=False,
    train_eval_downsample=.2)
dl = iter(data['train'])

# model

# model = Alphafold2(
#     dim=256,
#     depth=1,
#     heads=8,
#     dim_head=64
# ).to(DEVICE)

SRC_VOCAB_SIZE = 21  # number of amino acids
TGT_VOCAB_SIZE = 3  # backbone torsion angle
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
EMB_SIZE = 512
NUM_HEAD = 8
FFN_HID_DIM = 512
transformer = Seq2SeqTransformer(num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                                 emb_size=EMB_SIZE, src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE,
                                 dim_feedforward=FFN_HID_DIM, num_head=NUM_HEAD)

# optimizer

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)

# training loop
for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()
    train_loss = train_epoch(transformer, iter(data['train']), optimizer)
    end_time = time.time()
    valid_count = 0
    for split in scn.utils.download.VALID_SPLITS:
        val_loss = evaluate(transformer, iter(data[f'{split}']))
        writer_valids[valid_count].add_scalar("loss", val_loss, epoch)
        writer_valids[valid_count].flush()
        valid_count += 1
    writer_train.add_scalar("loss", train_loss, epoch)
    writer_train.flush()
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
           f"Epoch time = {(end_time - start_time):.3f}s"))
print('train ended')
writer_train.close()
valid_count = 0
for split in scn.utils.download.VALID_SPLITS:
    writer_valids[valid_count].close()
    valid_count += 1
