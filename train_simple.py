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
import os
import matplotlib.pyplot as plt

# constants

DEVICE = None  # defaults to cuda if available, else cpu
NUM_EPOCHS = int(3e5)
NUM_BATCHES = int(1e5)
GRADIENT_ACCUMULATE_EVERY = 16
LEARNING_RATE = 3e-4
IGNORE_INDEX = 21
THRESHOLD_LENGTH = 50
BATCH_SIZE = 250

# transformer constants

SRC_VOCAB_SIZE = 22  # number of amino acids + padding 21
TGT_VOCAB_SIZE = 3  # backbone torsion angle
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
EMB_SIZE = 256
NUM_HEAD = 8
FFN_HID_DIM = 128
LOSS_WITHOUT_PADDING = False

MODEL_PATH = f"model/model_t{THRESHOLD_LENGTH}_b{BATCH_SIZE}_e{NUM_ENCODER_LAYERS}_d{NUM_DECODER_LAYERS}_em{EMB_SIZE}_h{NUM_HEAD}_fh{FFN_HID_DIM}.pt"
BEST_MODEL_PATH = MODEL_PATH
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
            f"{portion}: {total_entires - n_filtered_entries:.0f} out of {total_entires:.0f} ({(total_entires - n_filtered_entries) / total_entires:.1%})"
            f" training set entries were included if sequence length <= {seq_length_threshold}")
    raw_data[portion] = new_data
    return raw_data


def create_mask(src, tgt):
    src_padding_mask = (src == IGNORE_INDEX).transpose(0, 1)
    tgt_padding_mask = (tgt == IGNORE_INDEX).transpose(0, 1)
    return src_padding_mask, tgt_padding_mask


def train_epoch(model, train_iter, optimizer_, epoch):
    model.train()
    losses = 0
    radian_diffs = torch.zeros(THRESHOLD_LENGTH*TGT_VOCAB_SIZE*BATCH_SIZE).to(DEVICE)
    logits_avg = torch.zeros(THRESHOLD_LENGTH * TGT_VOCAB_SIZE * BATCH_SIZE).to(DEVICE)
    angs_avg = torch.zeros(THRESHOLD_LENGTH * TGT_VOCAB_SIZE * BATCH_SIZE).to(DEVICE)
    for idx, (batch) in enumerate(train_iter):
        seq, coords, angs, mask = batch.seqs, batch.crds, batch.angs, batch.msks

        b, l, _ = seq.shape

        # prepare mask, labels

        seq, coords, angs, mask = seq.argmax(dim=-1).to(DEVICE), coords.to(DEVICE), angs.to(DEVICE), mask.to(
            DEVICE).bool()
        seq = F.pad(seq, (0, THRESHOLD_LENGTH - l), value=IGNORE_INDEX)
        coords = rearrange(coords, 'b (l c) d -> b l c d', l=l)
        if not LOSS_WITHOUT_PADDING:
            angs = F.pad(angs, (0, 0, 0, THRESHOLD_LENGTH - l), value=0)
        # angs = rearrange(angs, 'b l c -> b (l c)', l=THRESHOLD_LENGTH)
        mask = F.pad(mask, (0, THRESHOLD_LENGTH - l), value=False)

        # discretized_distances = get_bucketed_distance_matrix(coords[:, :, 1], mask, DISTOGRAM_BUCKETS, IGNORE_INDEX)
        src_padding_mask, tgt_padding_mask = create_mask(seq, seq)

        # predict

        logits = model(seq, seq, src_mask=mask,
                       tgt_mask=mask, src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)

        optimizer_.zero_grad()

        mask1= mask.unsqueeze(2).expand(-1, -1, 3)
        angs1 = torch.acos(torch.zeros(1)).item() * 4 * \
                          (angs < -torch.acos(torch.zeros(1)).item() * 1.5) +\
                          angs

        angs2 = mask1 * angs1
        logits2 = mask1 * logits
        angs3 = angs2.reshape(-1, angs2.shape[-1])
        logits3 = logits2.reshape(-1, logits2.shape[-1])

        # loss
        if LOSS_WITHOUT_PADDING:
            loss_ = loss_fn(logits[:, :l, :].reshape(-1, logits.shape[-1]), angs.reshape(-1, angs.shape[-1]))
            diff = logits[:, :l, :].reshape(-1, logits.shape[-1]) - angs.reshape(-1, angs.shape[-1])
        else:
            loss_ = loss_fn(logits3, angs3)
            diff = logits3 - angs3
        radian_diff = torch.rad2deg(diff).reshape(-1)
        radian_diffs += abs(radian_diff)
        logits_avg += abs(torch.rad2deg(logits3)).reshape(-1)
        angs_avg += abs(torch.rad2deg(angs3)).reshape(-1)

        # plt.plot(logits3.tolist(), label='logits')
        if idx == 0 and epoch % 10 == 0:
            plt.clf()
            plt.plot(angs3[:, 0:1].reshape(-1)[0:THRESHOLD_LENGTH].tolist(), label='phi')
            plt.plot(logits3[:, 0:1].reshape(-1)[0:THRESHOLD_LENGTH].tolist(), label='phi_logit')
            plt.ylabel('angles')
            plt.legend()
            plt.savefig(f"./graph/train1_{epoch}_phi.png")
            plt.clf()
            plt.plot(angs3[:, 1:2].reshape(-1)[0:THRESHOLD_LENGTH].tolist(), label='psi')
            plt.plot(logits3[:, 1:2].reshape(-1)[0:THRESHOLD_LENGTH].tolist(), label='psi_logit')
            plt.ylabel('angles')
            plt.legend()
            plt.savefig(f"./graph/train1_{epoch}_psi.png")
            plt.clf()
            plt.plot(angs3[:, 2:3].reshape(-1)[0:THRESHOLD_LENGTH].tolist(), label='omega')
            plt.plot(logits3[:, 2:3].reshape(-1)[0:THRESHOLD_LENGTH].tolist(), label='omega_logit')
            plt.ylabel('angles')
            plt.legend()
            plt.savefig(f"./graph/train1_{epoch}_omega.png")
        # plt.plot(diff.tolist())


        loss_.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer_.step()
        losses += loss_.item()
    radian_diffs = radian_diffs / len(train_iter)
    logits_avg = logits_avg / len(train_iter)
    angs_avg = angs_avg / len(train_iter)
    # diff_dict = {str(i): string for i, string in enumerate(radian_diffs.tolist())}
    # writer_train.add_scalars("train", diff_dict, epoch)
    if epoch % 10 == 0:
        plt.clf()
        plt.plot(torch.mean(radian_diffs.reshape(THRESHOLD_LENGTH*TGT_VOCAB_SIZE, -1), 1).tolist(), label='diff')
        plt.plot(torch.mean(logits_avg.reshape(THRESHOLD_LENGTH * TGT_VOCAB_SIZE, -1), 1).tolist(), label='logit')
        plt.plot(torch.mean(angs_avg.reshape(THRESHOLD_LENGTH * TGT_VOCAB_SIZE, -1), 1).tolist(), label='ang')
        plt.ylabel('angles')
        plt.legend()
        plt.savefig(f"./graph/train_{epoch}.png")
    return losses / len(train_iter)


def evaluate(model, val_iter):
    model.eval()
    losses = 0
    radian_diffs = None  # torch.zeros(THRESHOLD_LENGTH * TGT_VOCAB_SIZE * BATCH_SIZE).to(DEVICE)
    for idx, (batch) in (enumerate(val_iter)):
        seq, coords, angs, mask = batch.seqs, batch.crds, batch.angs, batch.msks

        b, l, _ = seq.shape
        if radian_diffs is None:
            radian_diffs = torch.zeros(THRESHOLD_LENGTH * TGT_VOCAB_SIZE * b).to(DEVICE)
        # prepare mask, labels

        seq, coords, angs, mask = seq.argmax(dim=-1).to(DEVICE), coords.to(DEVICE), angs.to(DEVICE), mask.to(
            DEVICE).bool()
        seq = F.pad(seq, (0, THRESHOLD_LENGTH - l), value=IGNORE_INDEX)
        coords = rearrange(coords, 'b (l c) d -> b l c d', l=l)
        if not LOSS_WITHOUT_PADDING:
            angs = F.pad(angs, (0, 0, 0, THRESHOLD_LENGTH - l), value=0)
        # angs = rearrange(angs, 'b l c -> b (l c)', l=THRESHOLD_LENGTH)
        mask = F.pad(mask, (0, THRESHOLD_LENGTH - l), value=False)

        # discretized_distances = get_bucketed_distance_matrix(coords[:, :, 1], mask, DISTOGRAM_BUCKETS, IGNORE_INDEX)
        src_padding_mask, tgt_padding_mask = create_mask(seq, seq)

        # predict

        logits = model(seq, seq, src_mask=mask,
                       tgt_mask=mask, src_padding_mask=src_padding_mask,
                       tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)

        angs_correction = torch.acos(torch.zeros(1)).item() * 4 * \
                          (angs.reshape(-1, angs.shape[-1]) < -torch.acos(torch.zeros(1)).item() * 1.5) + \
                          angs.reshape(-1, angs.shape[-1])

        # loss
        if LOSS_WITHOUT_PADDING:
            loss_ = loss_fn(logits[:, :l, :].reshape(-1, logits.shape[-1]), angs.reshape(-1, angs.shape[-1]))
            diff = logits[:, :l, :].reshape(-1, logits.shape[-1]) - angs.reshape(-1, angs.shape[-1])
        else:
            loss_ = loss_fn(logits.reshape(-1, logits.shape[-1]), angs_correction)
            diff = logits.reshape(-1, logits.shape[-1]) - angs_correction
        radian_diff = torch.rad2deg(diff).reshape(-1)
        radian_diffs += abs(radian_diff)

        losses += loss_.item()
    radian_diffs = radian_diffs / len(val_iter)
    # diff_dict = {str(i): string for i, string in enumerate(radian_diffs.tolist())}
    # writer_train.add_scalars("train", diff_dict, epoch)
    # plt.plot(torch.mean(radian_diffs.reshape(THRESHOLD_LENGTH * TGT_VOCAB_SIZE, -1), 1).tolist())
    # plt.ylabel('angles')
    # plt.savefig("valid.png")
    return losses / len(val_iter)


# get data

raw_data = scn.load(
    casp_version=12,
    thinning=30,
    batch_size=BATCH_SIZE,
    dynamic_batching=False
)

filtered_raw_data = filter_dictionary_by_seq_length(raw_data, THRESHOLD_LENGTH, "train")
writer_train = SummaryWriter("runs/train")
# writer_train_eval = SummaryWriter("runs/train_eval")
writer_valid = SummaryWriter("runs/validation")
# writer_valids = []
for split in scn.utils.download.VALID_SPLITS:
    filtered_raw_data = filter_dictionary_by_seq_length(filtered_raw_data, THRESHOLD_LENGTH, f'{split}')
#     writer_valids.append(SummaryWriter(f"runs/{split}"))
data = prepare_dataloaders(
    filtered_raw_data,
    aggregate_model_input=True,
    batch_size=BATCH_SIZE,
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

#
transformer = Seq2SeqTransformer(num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                                 emb_size=EMB_SIZE, src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE,
                                 dim_feedforward=FFN_HID_DIM, num_head=NUM_HEAD, activation='gelu', max_len=5000)

# optimizer

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)
# optimizer = torch.optim.RMSprop(
#     transformer.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False
# )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2000, verbose=True, factor=0.75)

prev_epoch = 0


def restore_model(model_path, model, optimizer_):
    prev_epoch_ = 0
    loss_ = 1e10
    valid_loss_ = 1e10
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        prev_epoch_ = checkpoint['epoch']
        loss_ = checkpoint['loss']
        if 'valid_loss' in checkpoint:
            valid_loss_ = checkpoint['valid_loss']
        print(f"restore checkpoint. Epoch: {prev_epoch_}, loss: {loss_:.3f}, valid_loss: {valid_loss_:.3f}")
    return prev_epoch_, loss_, valid_loss_


prev_epoch, loss, valid_loss = restore_model(MODEL_PATH, transformer, optimizer)
# training loop
best_valid = valid_loss if valid_loss < 1e10 else 1e10
restore_epoch = 10
for epoch in range(prev_epoch + 1, NUM_EPOCHS + 1):
    if epoch % restore_epoch == 0:
        restore_model(BEST_MODEL_PATH, transformer, optimizer)
    start_time = time.time()
    train_loss = train_epoch(transformer, iter(data['train']), optimizer, epoch)
    end_time = time.time()
    #    train_eval_loss = evaluate(transformer, iter(data['train-eval']))
    valid_count = 0
    val_loss_sum = 0
    for split in scn.utils.download.VALID_SPLITS:
        val_loss = evaluate(transformer, iter(data[f'{split}']))
        # writer_valids[valid_count].add_scalar("loss", val_loss, epoch)
        # writer_valids[valid_count].flush()
        # print(f"Epoch: {epoch}, {split} loss: {val_loss:.3f}")
        valid_count += 1
        val_loss_sum += val_loss
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, val loss: {val_loss_sum / valid_count:.3f}, "
           f"Epoch time = {(end_time - start_time):.3f}s"))
    writer_train.add_scalar("loss", train_loss, epoch)
    writer_train.flush()
    writer_valid.add_scalar("loss", val_loss_sum / valid_count, epoch)
    writer_valid.flush()
    # writer_train_eval.add_scalar("loss", train_eval_loss, epoch)
    # writer_train_eval.flush()
    scheduler.step(val_loss_sum / valid_count)
    torch.save({
        'epoch': epoch,
        'model_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': train_loss,
        'valid_loss': val_loss_sum / valid_count,
    }, MODEL_PATH)
    if val_loss_sum / valid_count < best_valid:
        best_valid = val_loss_sum / valid_count
        BEST_MODEL_PATH = f"model/model_t{THRESHOLD_LENGTH}_b{BATCH_SIZE}_e{NUM_ENCODER_LAYERS}_d{NUM_DECODER_LAYERS}_em{EMB_SIZE}_h{NUM_HEAD}_fh{FFN_HID_DIM}_{epoch}_{best_valid:.3f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
            'valid_loss': best_valid,
        }, BEST_MODEL_PATH)
        print(f"new best checkpoint. Epoch: {epoch}, loss: {train_loss:.3f}, valid_loss: {best_valid:.3f}")
print('train ended')
writer_train.close()
writer_valid.close()
# valid_count = 0
# for split in scn.utils.download.VALID_SPLITS:
#     writer_valids[valid_count].close()
#     valid_count += 1
