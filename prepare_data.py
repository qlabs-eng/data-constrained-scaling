"""
Tokenize FineWeb sample-10BT (GPT-2 BPE, every document prefixed with <|endoftext|>)
into three disjoint splits, in dataset order:

    val.bin    the first 10M tokens (identical to Slowrun's val split)
    train.bin  starting at the first document boundary after val
    calib.bin  the last documents of sample-10BT, ~2.1M tokens

Only the start and the last shard of the dataset are tokenized.

Usage:
    python prepare_data.py
    python prepare_data.py --train-tokens 1_000_000_000
"""

import os
import argparse
from collections import deque

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

VAL_TOKENS = 10_000_000
CALIB_TOKENS = 1025 * 2048
LAST_SHARD = "sample/10BT/014_00000.parquet"

parser = argparse.ArgumentParser()
parser.add_argument("--train-tokens", type=int, default=400_000_000)
parser.add_argument("--data-dir", type=str, default="data")
args = parser.parse_args()
os.makedirs(args.data_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def tokenize(texts):
    return [np.array([eot] + t, dtype=np.uint16) for t in enc.encode_ordinary_batch(texts, num_threads=16)]


def batches(dataset, size=1024):
    texts = []
    for doc in dataset:
        texts.append(doc["text"])
        if len(texts) == size:
            yield texts
            texts = []
    if texts:
        yield texts


# val and train: the start of the dataset
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
val, train = [], []
pos, n_train = 0, 0
pbar = tqdm(total=VAL_TOKENS + args.train_tokens, unit="tok")
for texts in batches(ds):
    docs = tokenize(texts)
    for doc in docs:
        if pos < VAL_TOKENS:
            val.append(doc[:VAL_TOKENS - pos])
        elif n_train < args.train_tokens:
            train.append(doc[:args.train_tokens - n_train])
            n_train += len(train[-1])
        pos += len(doc)
    pbar.update(sum(map(len, docs)))
    if n_train >= args.train_tokens:
        break
pbar.close()
assert n_train == args.train_tokens, f"only found {n_train:,} train tokens"
np.concatenate(val).tofile(os.path.join(args.data_dir, "val.bin"))
np.concatenate(train).tofile(os.path.join(args.data_dir, "train.bin"))

# calib: every document in the last CALIB_TOKENS tokens of the dataset that starts there
ds = load_dataset("HuggingFaceFW/fineweb", data_files=LAST_SHARD, split="train", streaming=True)
tail = deque(maxlen=10_000)
for doc in tqdm(ds, desc="last shard"):
    tail.append(doc["text"])
calib, n_calib = [], 0
for doc in reversed(tokenize(list(tail))):
    if n_calib + len(doc) > CALIB_TOKENS:
        break
    calib.append(doc)
    n_calib += len(doc)
np.concatenate(calib[::-1]).tofile(os.path.join(args.data_dir, "calib.bin"))

for split in ("val", "train", "calib"):
    n = os.path.getsize(os.path.join(args.data_dir, f"{split}.bin")) // 2
    print(f"{split}.bin: {n:,} tokens")
