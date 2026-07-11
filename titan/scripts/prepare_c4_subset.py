#!/usr/bin/env python
"""Prepare a small local C4 subset for the Level 1 debug labs.

TorchTitan's built-in ``c4_test`` dataset points at ``tests/assets/c4_test`` which is
NOT shipped in the pip wheel, and the full ``c4`` config streams ``allenai/c4`` from the
Hub (blocked by ``HF_HUB_OFFLINE=1``). This script extracts the first ``--n`` documents
from the Kempner-staged **real C4** shards into a local ``train.jsonl`` (a ``text`` column)
that TorchTitan's ``c4_test`` loader reads via ``load_dataset(dir, split="train")``:

    python scripts/prepare_c4_subset.py            # -> assets/c4_subset/train.jsonl (3000 docs)

Then the training labs pass:

    --training.dataset=c4_test --training.dataset_path=assets/c4_subset

Run once before the training labs. The output dir is git-ignored (real web text, not vendored).
"""
import argparse
import glob
import gzip
import json
import os

TESTBED = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/c4_original/raw/c4_en_train"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "c4_subset")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000, help="number of documents to extract")
    ap.add_argument("--src", default=TESTBED, help="dir of c4-train-*.json.gz shards")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dataset dir")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.src, "c4-train-*.json.gz")))
    if not shards:
        raise SystemExit(f"No C4 shards found under {args.src}")

    out_path = os.path.join(out_dir, "train.jsonl")
    written = 0
    with open(out_path, "w") as fout:
        for shard in shards:
            with gzip.open(shard, "rt") as fin:
                for line in fin:
                    rec = json.loads(line)
                    fout.write(json.dumps({"text": rec["text"]}) + "\n")
                    written += 1
                    if written >= args.n:
                        break
            if written >= args.n:
                break

    size_mb = round(os.path.getsize(out_path) / 1e6, 1)
    print(f"wrote {written} docs ({size_mb} MB) -> {out_path}")
    print(f"use: --training.dataset=c4_test --training.dataset_path={out_dir}")


if __name__ == "__main__":
    main()
