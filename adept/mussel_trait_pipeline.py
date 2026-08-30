#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mussel_trait_pipeline.py -- shared readers and basis transfers for the mussel
material: a phased-VCF parser, identifier matching between a VCF and a
phenotype table, sex-column detection, and the NOIA / fixed-frequency GRM
helpers that move a reference-population basis onto a candidate population.
Imported by the mussel scripts that need those pieces without the full driver.
"""
from __future__ import annotations
import gzip, json, os, re, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pig_epik_pipeline as P
import pig_epistasis_pipeline as Q

DERIVED_TRAITS = ("phe8", "phe12", "phe13", "phe14")

def say(*a):
    print(*a, flush=True)

def jdump(o, p):
    with open(p, "w") as f:
        json.dump(o, f, indent=2, default=str)

def read_phased_vcf(path):
    opener = gzip.open if path.endswith(".gz") else open
    samples = None
    chrom, pos, names = [], [], []
    rows = []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                parts = line.rstrip("\n").split("\t")
                samples = parts[9:]
                continue
            if samples is None:
                raise ValueError(f"{path}: no #CHROM header")
            parts = line.rstrip("\n").split("\t")
            c, p_, vid, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            if "," in alt:
                continue
            fmt = parts[8].split(":")
            gt_idx = fmt.index("GT") if "GT" in fmt else 0
            gts = np.full(len(samples), np.nan)
            for i, cell in enumerate(parts[9:]):
                gt = cell.split(":")[gt_idx]
                a = re.split(r"[|/]", gt)
                if len(a) != 2 or "." in a:
                    continue
                gts[i] = int(a[0]) + int(a[1])
            chrom.append(c); pos.append(int(p_))
            names.append(vid if vid not in (".", "") else f"{c}:{p_}")
            rows.append(gts)
    X = np.array(rows, dtype=np.float64).T
    n_missing = int(np.isnan(X).sum())
    if n_missing:
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_mean, idx[1])
        say(f"  [VCF] {path}: filled {n_missing} missing genotypes")
    return X, np.array(chrom), np.array(pos, dtype=int), names, samples

def detect_id_col(df, id_hint_names=("id", "sample", "animal", "ind", "iid")):
    for c in df.columns:
        if str(c).strip().lower() in id_hint_names:
            return c
    return df.columns[0]


def _collapse_doubled(s):
    """'c1_c1' -> 'c1'. Only collapses when the two halves either side of the
    underscore are identical, so an ordinary identifier such as 'sample_01' is
    left alone."""
    parts = s.split("_")
    if len(parts) == 2 and parts[0] and parts[0] == parts[1]:
        return parts[0]
    return s


def match_ids_to_pheno(vcf_ids, pheno_df, label="reference population",
                       min_match_frac=0.5):
    """
    Match VCF sample names to the index of a phenotype table. A 100% match is
    not required: an individual that is genotyped but not yet phenotyped is a
    normal occurrence. What matters is that every unmatched identifier is
    listed in full rather than truncated, that downstream work continues on the
    intersection, and that a systematic naming mismatch is not passed off as
    "some individuals have no phenotype": if the matched fraction falls below
    min_match_frac the run aborts.

    Returns (keep_mask, matched_pheno_ids, strategy_name):
      keep_mask          boolean array of len(vcf_ids); True where the VCF
                         individual was found in the phenotype table. Use it to
                         subset the rows of the genotype matrix.
      matched_pheno_ids  the pheno_df.index labels corresponding to the True
                         positions, length keep_mask.sum(), ready for .loc
      strategy_name      which transformation was used, for the log
    """
    pheno_ids = set(str(x) for x in pheno_df.index)
    pheno_ids_lower = {p.lower() for p in pheno_ids}
    lower_to_orig = {str(p).lower(): p for p in pheno_df.index}
    vcf_ids = [str(x) for x in vcf_ids]
    n = len(vcf_ids)

    strategies = [
        ("as-is", lambda s: s, pheno_ids),
        ("stripped", lambda s: s.strip(), pheno_ids),
        ("collapse doubled (c1_c1 -> c1)", _collapse_doubled, pheno_ids),
        ("collapse doubled + stripped", lambda s: _collapse_doubled(s.strip()), pheno_ids),
        ("case-insensitive", lambda s: s.strip().lower(), pheno_ids_lower),
    ]

    best = None  # (hits, name, mask, mapped)
    for name, fn, target_set in strategies:
        mapped = [fn(v) for v in vcf_ids]
        mask = np.array([m in target_set for m in mapped])
        hits = int(mask.sum())
        if best is None or hits > best[0]:
            best = (hits, name, mask, mapped)

    hits, strategy, mask, mapped = best
    frac = hits / n if n else 0.0
    say(f"  [id match] best strategy '{strategy}': {hits}/{n} matched ({frac:.1%})")

    unmatched = [vcf_ids[i] for i in range(n) if not mask[i]]
    if unmatched:
        if len(unmatched) <= 50:
            say(f"  [id match] {len(unmatched)} unmatched (listed in full): {unmatched}")
        else:
            say(f"  [id match] {len(unmatched)} unmatched (showing the first 30)"
               f": {unmatched[:30]} ... and {len(unmatched)-30} more")

    if frac < min_match_frac:
        say(f"\n  A match rate of {frac:.1%} is below the {min_match_frac:.0%} "
            f"threshold. This is unlikely to be a few unphenotyped individuals "
            f"and far more likely a naming mismatch; dropping this many "
            f"individuals silently is not acceptable.")
        say(f"  example VCF sample names (first 5): {vcf_ids[:5]}")
        say(f"  example phenotype index labels (first 5): {list(pheno_df.index[:5])}")
        raise SystemExit(f"{label}: identifier match rate too low ({frac:.1%}). "
                         f"Read the full unmatched list above rather than "
                         f"guessing another transformation.")

    matched_pheno_ids = []
    for i in range(n):
        if mask[i]:
            m = mapped[i]
            matched_pheno_ids.append(lower_to_orig.get(m, m)
                                     if strategy == "case-insensitive" else m)
    return mask, matched_pheno_ids, strategy


def detect_sex_col(df, exclude):
    # Chinese sex tokens are kept because real input files use them.
    male_tok = {"m", "male", "1", "\u516c"}
    female_tok = {"f", "female", "2", "\u6bcd"}
    for c in df.columns:
        if c == exclude:
            continue
        vals = set(str(v).strip().lower() for v in df[c].dropna().unique())
        if vals and vals <= (male_tok | female_tok):
            return c
    raise ValueError(f"cannot identify sex column from {list(df.columns)}")

def normalize_sex(series):
    # Chinese sex tokens are kept because real input files use them.
    male_tok = {"m", "male", "1", "\u516c"}
    return series.astype(str).str.strip().str.lower().isin(male_tok)

def noia_fit(X):
    pAA = np.mean(X == 0, 0); pAa = np.mean(X == 1, 0); paa = np.mean(X == 2, 0)
    mu = pAa + 2 * paa
    Ha = X - mu
    v = pAA * mu**2 + pAa * (1 - mu)**2 + paa * (2 - mu)**2
    mono = v < 1e-12
    H = (X == 1).astype(float)
    beta_d = np.where(mono, 0, (pAa * (1 - mu)) / np.where(mono, 1, v))
    Hd = H - pAa - beta_d * Ha
    Hd[:, mono] = 0.
    return Ha, Hd, mu, pAa, beta_d

def apply_noia_basis(X_new, mu, pAa, beta_d):
    Ha_new = X_new - mu
    H_new = (X_new == 1).astype(float)
    Hd_new = H_new - pAa - beta_d * Ha_new
    return Ha_new, Hd_new

def grm_a_fixed_freq(X, p_ref):
    Z = X - 2 * p_ref[None, :]
    return (Z @ Z.T) / (2 * np.sum(p_ref * (1 - p_ref)) + 1e-12)
