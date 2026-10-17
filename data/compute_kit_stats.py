"""Recompute the z-normalization statistics (mean/std) of KIT-ML from its train split.

Writes:
  - <data_root>/Mean.npy, Std.npy   (overwritten in place)

Usage: python data/compute_kit_stats.py
"""
import argparse
import os
from os.path import join as pjoin

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./dataset/KIT-ML/")
    parser.add_argument("--split", default="train.txt")
    parser.add_argument("--min_len", type=int, default=24, help="skip motions shorter than this (matches the eval dataset filter)")
    args = parser.parse_args()

    with open(pjoin(args.data_root, args.split)) as f:
        ids = [line.strip() for line in f if line.strip()]

    n, s1, s2, used = 0, 0.0, 0.0, 0
    for name in ids:
        path = pjoin(args.data_root, "new_joint_vecs", name + ".npy")
        if not os.path.exists(path):
            print("skip (missing):", name)
            continue
        x = np.load(path).astype(np.float64)
        if len(x) < args.min_len:
            continue
        n += len(x)
        s1 = s1 + x.sum(0)
        s2 = s2 + (x ** 2).sum(0)
        used += 1

    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
    assert (std > 0).all(), "zero-variance dimension found"
    print(f"{used}/{len(ids)} motions, {n} frames, dim {mean.shape[0]}")

    np.save(pjoin(args.data_root, "Mean.npy"), mean)
    np.save(pjoin(args.data_root, "Std.npy"), std)

    print("saved to", args.data_root)


if __name__ == "__main__":
    main()
