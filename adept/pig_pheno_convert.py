#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCTA-style per-trait .phe files (FID IID value, no header) -> one wide CSV
with columns id,<trait1>,<trait2>,... -- the format mateagent.py's --pheno
expects.

Pig's phenotypes live as one file per trait (bf.phe, lmd.phe, ...), which is
GCTA's native convention. mateagent.py's ingest stage expects the mussel-style
wide format (one id column, one column per trait) instead. This converts
without touching either convention's own code -- pig_epistasis_pipeline.py
keeps reading the per-trait files exactly as before.

    python pig_pheno_convert.py --traits bf lmd lmp ltn rtn tpd ttn \
        --out pig_pheno_wide.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def read_gcta_phe(path):
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["fid", "iid", "value"])
    return df.set_index("iid")["value"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", nargs="+", required=True)
    ap.add_argument("--suffix", default=".phe")
    ap.add_argument("--out", default="pig_pheno_wide.csv")
    a = ap.parse_args()

    series = {}
    for t in a.traits:
        p = f"{t}{a.suffix}"
        if not os.path.exists(p):
            print(f"[skip] {t}: {p} not found")
            continue
        series[t] = read_gcta_phe(p)
        print(f"  {t}: {series[t].notna().sum()} individuals")

    if not series:
        raise SystemExit("no trait files found")

    wide = pd.DataFrame(series)
    wide.index.name = "id"
    wide = wide.reset_index()
    wide.to_csv(a.out, index=False)
    print(f"\n-> {a.out}  ({len(wide)} individuals, "
          f"{len(series)} trait columns)")
    print("  NOTE: individuals not phenotyped for a given trait are NaN in "
          "that column, exactly as in the per-trait files -- mateagent's "
          "min_n / finite-value filtering handles this per trait already.")


if __name__ == "__main__":
    main()
