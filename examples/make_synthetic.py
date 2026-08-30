#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 make_synthetic.py -- a small synthetic dataset so the pipeline can be run
                      end to end without any restricted data
================================================================================

 Writes PLINK .bed/.bim/.fam, a GCTA-style .phe, and a sex file. Two modes:

   --mode outbred   a structured, outcrossing population; dominance and
                    epistasis modules are available
   --mode inbred    an essentially homozygous panel; capability detection
                    should BLOCK the dominance module and the dominance-derived
                    epistatic families

 The phenotype carries a polygenic additive background plus, in outbred mode,
 a dominance component and a small number of concentrated cross-chromosome
 interacting pairs. The true pairs are written to <out>_truth.json so a run
 can be checked against them.

 This dataset is far smaller than any material in the paper and exists to
 verify that the code runs and that capability detection behaves, not to
 reproduce any published number. Detection power at n = 400 is low by
 construction.

     python make_synthetic.py --out demo --mode outbred --n 400 --m 3000
================================================================================
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def write_plink(prefix, X, chrom, pos, ids, sex_codes):
    n, m = X.shape
    with open(prefix + ".bim", "w") as f:
        for j in range(m):
            f.write(f"{chrom[j]}\tsnp{j + 1}\t0\t{pos[j]}\tA\tG\n")
    with open(prefix + ".fam", "w") as f:
        for i in range(n):
            f.write(f"FAM\t{ids[i]}\t0\t0\t{sex_codes[i]}\t-9\n")
    # PLINK 2-bit codes, SNP-major: 00 hom A1 (dosage 2), 10 het (1),
    # 11 hom A2 (0). 01 is missing and is never written here.
    code = np.select([X == 2, X == 1], [0b00, 0b10], default=0b11
                     ).astype(np.uint8)
    bpsnp = (n + 3) // 4
    out = np.zeros((m, bpsnp), dtype=np.uint8)
    for b in range(4):
        idx = np.arange(b, n, 4)
        if not len(idx):
            continue
        out[:, :len(idx)] |= (code[idx, :].T << (2 * b)).astype(np.uint8)
    with open(prefix + ".bed", "wb") as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        f.write(out.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo")
    ap.add_argument("--mode", choices=["outbred", "inbred"], default="outbred")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--m", type=int, default=3000)
    ap.add_argument("--n_chr", type=int, default=10)
    ap.add_argument("--h2_add", type=float, default=0.35)
    ap.add_argument("--h2_dom", type=float, default=0.08)
    ap.add_argument("--h2_epi", type=float, default=0.12)
    ap.add_argument("--n_pairs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    per_chr = a.m // a.n_chr
    m = per_chr * a.n_chr
    chrom = np.repeat(np.arange(1, a.n_chr + 1), per_chr)
    pos = np.tile((np.arange(per_chr) + 1) * 50_000, a.n_chr)

    # two subpopulations with shifted frequencies, so the additive kernel has
    # real structure rather than being white noise
    p0 = rng.uniform(0.15, 0.85, m)
    shift = rng.uniform(-0.10, 0.10, m)
    n1 = a.n // 2
    X = np.empty((a.n, m))
    for k in range(m):
        pa = np.clip(p0[k] + shift[k], 0.05, 0.95)
        pb = np.clip(p0[k] - shift[k], 0.05, 0.95)
        if a.mode == "inbred":
            # draw a haplotype and double it: heterozygotes essentially absent
            X[:n1, k] = 2 * rng.binomial(1, pa, n1)
            X[n1:, k] = 2 * rng.binomial(1, pb, a.n - n1)
        else:
            X[:n1, k] = rng.binomial(2, pa, n1)
            X[n1:, k] = rng.binomial(2, pb, a.n - n1)

    if a.mode == "inbred":
        # a handful of residual heterozygotes, as in a real selfed panel
        for _ in range(max(1, int(0.0005 * a.n * m))):
            X[rng.integers(a.n), rng.integers(m)] = 1

    p = X.mean(0) / 2
    W = X - 2 * p                                     # additive code
    H = np.where(X == 1, 2 * p * (1 - p),
                 np.where(X == 0, -2 * p ** 2, -2 * (1 - p) ** 2))

    def z(v):
        v = v - v.mean()
        s = v.std()
        return v / s if s > 1e-12 else v

    causal = rng.choice(m, size=min(200, m), replace=False)
    g_add = z(W[:, causal] @ rng.normal(size=len(causal)))

    if a.mode == "inbred":
        h2_dom = 0.0
        g_dom = np.zeros(a.n)
    else:
        h2_dom = a.h2_dom
        dcaus = rng.choice(m, size=min(100, m), replace=False)
        g_dom = z(H[:, dcaus] @ rng.normal(size=len(dcaus)))

    truth_pairs = []
    g_epi = np.zeros(a.n)
    if a.h2_epi > 0 and a.n_pairs > 0:
        for _ in range(a.n_pairs):
            while True:
                i, j = rng.integers(m), rng.integers(m)
                if chrom[i] != chrom[j]:
                    break
            truth_pairs.append([int(i), int(j)])
            g_epi = g_epi + z(W[:, i] * W[:, j]) * rng.choice([-1.0, 1.0])
        g_epi = z(g_epi)

    h2e = a.h2_epi if truth_pairs else 0.0
    resid = max(1e-6, 1.0 - a.h2_add - h2_dom - h2e)
    y = (np.sqrt(a.h2_add) * g_add + np.sqrt(h2_dom) * g_dom
         + np.sqrt(h2e) * g_epi + np.sqrt(resid) * rng.normal(size=a.n))

    ids = [f"ID{i + 1:05d}" for i in range(a.n)]
    sex_codes = rng.integers(1, 3, a.n)
    write_plink(a.out, X.astype(int), chrom, pos, ids, sex_codes)

    with open(a.out + ".phe", "w") as f:
        for i in range(a.n):
            f.write(f"FAM {ids[i]} {y[i]:.6f}\n")
    with open(a.out + "_sex.csv", "w") as f:
        f.write("id,sex\n")
        for i in range(a.n):
            f.write(f"{ids[i]},{'M' if sex_codes[i] == 1 else 'F'}\n")

    truth = dict(mode=a.mode, n=a.n, m=m, seed=a.seed,
                 h2_add=a.h2_add, h2_dom=h2_dom, h2_epi=h2e,
                 mean_heterozygosity=float(np.mean(X == 1)),
                 true_pairs=[[f"snp{i + 1}", f"snp{j + 1}"]
                             for i, j in truth_pairs])
    with open(a.out + "_truth.json", "w") as f:
        json.dump(truth, f, indent=2)

    print(f"-> {a.out}.bed/.bim/.fam, {a.out}.phe, {a.out}_sex.csv, "
          f"{a.out}_truth.json")
    print(f"   {a.n} individuals x {m} markers, mode={a.mode}, "
          f"mean heterozygosity {truth['mean_heterozygosity']:.5f}")
    if a.mode == "inbred":
        print("   capability detection should BLOCK the dominance module "
              "on this file")


if __name__ == "__main__":
    main()
