import pickle
import string
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.linalg as LA
import prody
import torch
from Bio import SeqIO
from einops import repeat
from sidechainnet.utils.measure import get_seq_coords_and_angles
from sidechainnet.utils.sequence import ProteinVocabulary
from torch.utils.data import DataLoader, Dataset
from alphafold2_pytorch.constants import DISTOGRAM_BUCKETS
from tqdm import tqdm

try:
    import pytorch_lightning as pl

    LightningDataModule = pl.LightningDataModule
except ImportError:
    LightningDataModule = object

CACHE_PATH = Path("~/.cache/alphafold2_pytorch").expanduser()
DATA_DIR = CACHE_PATH / "trrosetta" / "trrosetta"
URL = "http://s3.amazonaws.com/proteindata/data_pytorch/trrosetta.tar.gz"

REMOVE_KEYS = dict.fromkeys(string.ascii_lowercase)
REMOVE_KEYS["."] = None
REMOVE_KEYS["*"] = None
translation = str.maketrans(REMOVE_KEYS)

DEFAULT_VOCAB = ProteinVocabulary()


def default_tokenize(seq: str) -> List[int]:
    return [DEFAULT_VOCAB[ch] for ch in seq]


def read_fasta(filename: str) -> List[Tuple[str, str]]:
    def remove_insertions(sequence: str) -> str:
        return sequence.translate(translation)

    return [
        (record.description, remove_insertions(str(record.seq)))
        for record in SeqIO.parse(filename, "fasta")
    ]


def read_pdb(pdb: str):
    ag = prody.parsePDB(pdb)
    for chain in ag.iterChains():
        angles, coords, seq = get_seq_coords_and_angles(chain)
        return angles, coords, seq


def download_file(url, filename=None, root=CACHE_PATH):
    import os
    import urllib

    root.mkdir(exist_ok=True, parents=True)
    filename = filename or os.path.basename(url)

    download_target = root / filename
    download_target_tmp = root / f"tmp.{filename}"

    if download_target.exists() and not download_target.is_file():
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if download_target.is_file():
        return download_target

    with urllib.request.urlopen(url) as source, open(
        download_target_tmp, "wb"
    ) as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    download_target_tmp.rename(download_target)
    return download_target


def get_or_download(url: str = URL):
    """
    download and extract trrosetta data
    """
    import tarfile

    file = CACHE_PATH / "trrosetta.tar.gz"
    dir = CACHE_PATH / "trrosetta"
    dir_temp = CACHE_PATH / "trrosetta_tmp"
    if dir.is_dir():
        print(f"Load cached data from {dir}")
        return dir

    if not file.is_file():
        print(f"Cache not found, download from {url} to {file}")
        download_file(url)

    print(f"Extract data from {file} to {dir}")
    with tarfile.open(file, "r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, dir_temp)

    dir_temp.rename(dir)

    return dir


def pad_sequences(sequences, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class TrRosettaDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        list_path: Path,
        tokenize: Callable[[str], List[int]],
        seq_pad_value: int = 20,
        random_sample_msa: bool = False,
        max_seq_len: int = 300,
        max_msa_num: int = 300,
        overwrite: bool = False,
    ):
        self.data_dir = data_dir
        self.file_list: List[Path] = self.read_file_list(data_dir, list_path)

        self.tokenize = tokenize
        self.seq_pad_value = seq_pad_value

        self.random_sample_msa = random_sample_msa
        self.max_seq_len = max_seq_len
        self.max_msa_num = max_msa_num

        self.overwrite = overwrite

    def __len__(self) -> int:
        return len(self.file_list)

    def read_file_list(self, data_dir: Path, list_path: Path):
        file_glob = (data_dir / "npz").glob("*.npz")
        files = set(list_path.read_text().split())
        if len(files) == 0:
            raise ValueError("Passed an empty split file set")

        file_list = [f for f in file_glob if f.name in files]
        if len(file_list) != len(files):
            num_missing = len(files) - len(file_list)
            raise FileNotFoundError(
                f"{num_missing} specified split files not found in directory"
            )

        return file_list

    def has_cache(self, index):
        if self.overwrite:
            return False

        path = (self.data_dir / "cache" / self.file_list[index].stem).with_suffix(
            ".pkl"
        )
        return path.is_file()

    def write_cache(self, index, data):
        path = (self.data_dir / "cache" / self.file_list[index].stem).with_suffix(
            ".pkl"
        )
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as file:
            pickle.dump(data, file)

    def read_cache(self, index):
        path = (self.data_dir / "cache" / self.file_list[index].stem).with_suffix(
            ".pkl"
        )
        with open(path, "rb") as file:
            return pickle.load(file)

    def __getitem__(self, index):
        if self.has_cache(index):
            item = self.read_cache(index)
        else:
            id = self.file_list[index].stem
            pdb_path = self.data_dir / "pdb" / f"{id}.pdb"
            msa_path = self.data_dir / "a3m" / f"{id}.a3m"
            _, msa = zip(*read_fasta(str(msa_path)))
            msa = np.array([np.array(list(seq)) for seq in msa])
            angles, coords, seq = read_pdb(str(pdb_path))
            seq = np.array(list(seq))
            coords = coords.reshape((coords.shape[0] // 14, 14, 3))
            dist = self.get_bucketed_distance(seq, coords, subset="ca")
            item = {
                "id": id,
                "seq": seq,
                "msa": msa,
                "coords": coords,
                "angles": angles,
                "dist": dist
            }
            self.write_cache(index, item)

        item["msa"] = self.sample(item["msa"], self.max_msa_num, self.random_sample_msa)
        item = self.crop(item, self.max_seq_len)
        return item

    def calc_cb(self, coord):
        N = coord[0]
        CA = coord[1]
        C = coord[2]

        b = CA - N
        c = C - CA
        a = np.cross(b, c)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        return CB

    def get_bucketed_distance(
        self, seq, coords, subset="ca", start=2, bins=DISTOGRAM_BUCKETS-1, step=0.5
    ):
        assert subset in ("ca", "cb")
        if subset == "ca":
            coords = coords[:, 1, :]
        elif subset == "cb":
            cb_coords = []
            for res, coord in zip(seq, coords):
                if res == "G":
                    cb = self.calc_cb(coord)
                    cb_coords.append(cb)
                else:
                    cb_coords.append(coord[4, :])
            coords = np.array(cb_coords)
        vcs = coords + np.zeros([coords.shape[0]] + list(coords.shape))
        vcs = vcs - np.swapaxes(vcs, 0, 1)
        distance_map = LA.norm(vcs, axis=2)
        mask = np.ones(distance_map.shape) - np.eye(distance_map.shape[0])
        low_pos = np.where(distance_map < start)
        high_pos = np.where(distance_map >= start + step * bins)

        mask[low_pos] = 0
        distance_map = (distance_map - start) // step
        distance_map[high_pos] = bins
        dist = (distance_map * mask).astype(int)
        return dist

    def crop(self, item, max_seq_len: int):
        seq_len = len(item["seq"])

        if seq_len <= max_seq_len or max_seq_len <= 0:
            return item

        start = 0
        end = start + max_seq_len

        item["seq"] = item["seq"][start:end]
        item["msa"] = item["msa"][:, start:end]
        item["coords"] = item["coords"][start:end]
        item["angles"] = item["angles"][start:end]
        item["dist"] = item["dist"][start:end, start:end]
        return item

    def sample(self, msa, max_msa_num: int, random: bool):
        num_msa, seq_len = len(msa), len(msa[0])

        if num_msa <= max_msa_num or max_msa_num <= 0:
            return msa

        if random:
            num_sample = max_msa_num - 1
            indices = np.random.choice(num_msa - 1, size=num_sample, replace=False) + 1
            indices = np.pad(indices, [1, 0], "constant")
            return msa[indices]
        else:
            return msa[:max_msa_num]

    def collate_fn(self, batch):
        b = len(batch)
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        id = batch["id"]
        seq = batch["seq"]
        msa = batch["msa"]
        coords = batch["coords"]
        angles = batch["angles"]
        dist = batch["dist"]

        lengths = torch.LongTensor([len(x[0]) for x in msa])
        depths = torch.LongTensor([len(x) for x in msa])
        max_len = lengths.max()
        max_depth = depths.max()

        seq = pad_sequences(
            [torch.LongTensor(self.tokenize(seq_)) for seq_ in seq], self.seq_pad_value,
        )

        msa = pad_sequences(
            [torch.LongTensor([self.tokenize(seq_) for seq_ in msa_]) for msa_ in msa],
            self.seq_pad_value,
        )

        coords = pad_sequences([torch.FloatTensor(x) for x in coords], 0.0)

        angles = pad_sequences([torch.FloatTensor(x) for x in angles], 0.0)

        dist = pad_sequences([torch.LongTensor(x) for x in dist], -100)

        mask = repeat(torch.arange(max_len), "l -> b l", b=b) < repeat(
            lengths, "b -> b l", l=max_len
        )
        msa_seq_mask = repeat(
            torch.arange(max_len), "l -> b s l", b=b, s=max_depth
        ) < repeat(lengths, "b -> b s l", s=max_depth, l=max_len)
        msa_depth_mask = repeat(
            torch.arange(max_depth), "s -> b s l", b=b, l=max_len
        ) < repeat(depths, "b -> b s l", s=max_depth, l=max_len)
        msa_mask = msa_seq_mask & msa_depth_mask

        return {
            "id": id,
            "seq": seq,
            "msa": msa,
            "coords": coords,
            "angles": angles,
            "mask": mask,
            "msa_mask": msa_mask,
            "dist": dist,
        }


class TrRosettaDataModule(LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--eval_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)

        parser.add_argument("--num_workers", type=int, default=0)

        parser.add_argument("--train_max_seq_len", type=int, default=256)
        parser.add_argument("--eval_max_seq_len", type=int, default=256)
        parser.add_argument("--test_max_seq_len", type=int, default=-1)

        parser.add_argument("--train_max_msa_num", type=int, default=256)
        parser.add_argument("--eval_max_msa_num", type=int, default=256)
        parser.add_argument("--test_max_msa_num", type=int, default=1000)

        parser.add_argument("--overwrite", dest="overwrite", action="store_true")

        return parser

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        test_batch_size: int = 1,
        num_workers: int = 0,
        train_max_seq_len: int = 256,
        eval_max_seq_len: int = 256,
        test_max_seq_len: int = -1,
        train_max_msa_num: int = 32,
        eval_max_msa_num: int = 32,
        test_max_msa_num: int = 64,
        tokenize: Callable[[str], List[int]] = default_tokenize,
        seq_pad_value: int = 20,
        overwrite: bool = False,
        **kwargs,
    ):
        super(TrRosettaDataModule, self).__init__()
        self.data_dir = Path(data_dir).expanduser().resolve()

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size

        self.num_workers = num_workers

        self.train_max_seq_len = train_max_seq_len
        self.eval_max_seq_len = eval_max_seq_len
        self.test_max_seq_len = test_max_seq_len

        self.train_max_msa_num = train_max_msa_num
        self.eval_max_msa_num = eval_max_msa_num
        self.test_max_msa_num = test_max_msa_num

        self.tokenize = tokenize
        self.seq_pad_value = seq_pad_value

        self.overwrite = overwrite

        get_or_download()

    def setup(self, stage: Optional[str] = None):
        self.train = TrRosettaDataset(
            self.data_dir,
            self.data_dir / "train_files.txt",
            self.tokenize,
            self.seq_pad_value,
            random_sample_msa=True,
            max_seq_len=self.train_max_seq_len,
            max_msa_num=self.train_max_msa_num,
            overwrite=self.overwrite,
        )

        self.val = TrRosettaDataset(
            self.data_dir,
            self.data_dir / "valid_files.txt",
            self.tokenize,
            self.seq_pad_value,
            random_sample_msa=False,
            max_seq_len=self.eval_max_seq_len,
            max_msa_num=self.eval_max_msa_num,
            overwrite=self.overwrite,
        )

        self.test = TrRosettaDataset(
            self.data_dir,
            self.data_dir / "valid_files.txt",
            self.tokenize,
            self.seq_pad_value,
            random_sample_msa=False,
            max_seq_len=self.test_max_seq_len,
            max_msa_num=self.test_max_msa_num,
            overwrite=self.overwrite,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.train.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.val.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.test.collate_fn,
            num_workers=self.num_workers,
        )


def test():
    dm = TrRosettaDataModule(train_batch_size=1, num_workers=4)
    dm.setup()

    for batch in dm.train_dataloader():
        print("id", batch["id"])
        print("seq", batch["seq"].shape, batch["seq"])
        print("msa", batch["msa"].shape, batch["msa"][..., :20])
        print("msa", batch["msa"].shape, batch["msa"][..., -20:])
        print("coords", batch["coords"].shape)
        print("angles", batch["angles"].shape)
        print("mask", batch["mask"].shape)
        print("msa_mask", batch["msa_mask"].shape)
        print("dist", batch["dist"].shape, batch["dist"])
        break


if __name__ == "__main__":
    test()
