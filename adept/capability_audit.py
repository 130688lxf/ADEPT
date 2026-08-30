#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 capability_audit.py -- what can this material support, and is any kernel a
                        disguised ridge penalty?
================================================================================

 Run this BEFORE fitting anything. It answers two questions that decide how
 the rest of the pipeline is allowed to behave, and it answers them from the
 genotypes alone.

 [1] CAPABILITY DETECTION (Methods 2.4)
     Mean heterozygosity, per-marker dominance-code variance, phase, recorded
     sex, and the presence of an independent candidate population determine
     which modules may be fitted. A module the material cannot support is
     BLOCKED and reported as blocked. It is never fitted and returned as a
     numerical zero, because those two outcomes mean opposite things: a block
     is a statement about the germplasm, a zero is a statement about the trait.
     In an inbred panel the dominance code has almost no variance, so a
     dominance term fitted there describes the panel. Crossing two inbred
     lines still yields a fully heterozygous F1 in which dominance is
     expressed; what is missing is any information with which to estimate it.

 [2] KERNEL DEGENERACY AUDIT (Methods 2.4, Results 3.3)
     A kernel close to a multiple of the identity matrix carries no covariance
     between individuals. Fitted as a random effect it acts as a ridge penalty:
     it stabilises accuracy across splits without describing any genetic
     relationship, and it can therefore pass a positive-fraction criterion
     while carrying no genetic information at all. Two diagnostics from the
     Methods are reported for every kernel:
         off_diag_ratio  mean |off-diagonal| / mean diagonal
         cor_with_I      correlation between the kernel and the identity
     A ratio near zero together with a correlation near one identifies a
     degenerate kernel, which must be excluded before any decision is taken.

     Eigenvalue summaries are also reported. Their definitions are printed
     with the table, because "effective rank" has several definitions in use
     and only numbers computed under the same definition are comparable.

 [3] KERNEL COLLINEARITY
     Off-diagonal correlations between every pair of kernels. Where these are
     high, variance components are not separately identified, so only
     out-of-sample prediction gain may be reported and variance fractions are
     descriptive at best.

     python capability_audit.py --bfile <plink_prefix> --out capability_pig
     python capability_audit.py --bfile <prefix> --sex cand_sex.csv \
         --candidates <breed_prefix> --vcf ref_all_phased.vcf --out capability
================================================================================
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pig_epik_pipeline as P
import pig_epistasis_pipeline as Q

HET_BLOCK_THRESHOLD = 0.02
DEGENERATE_RATIO = 0.02          # mean|off-diag| / mean diag below this
DEGENERATE_COR_I = 0.95          # and correlation with I above this


def say(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# genotype loading
# --------------------------------------------------------------------------
def load_plink(prefix):
    bim = P.read_bim(prefix)
    fam = P.read_fam(prefix)
    X = P.read_bed(prefix, len(fam), len(bim))
    miss = X < 0
    if miss.any():
        for c in np.where(miss.any(0))[0]:
            col = X[:, c]
            obs = col[col >= 0]
            col[col < 0] = obs.mean() if len(obs) else 0.0
            X[:, c] = col
    return (np.ascontiguousarray(X, dtype=np.float32),
            bim["chrom"].astype(str).values,
            bim["snp"].astype(str).values,
            fam["iid"].astype(str).values,
            fam["sex"].astype(str).values)


def phased_fraction(vcf_path, max_rows=2000):
    """Fraction of sampled genotype fields that use '|' rather than '/'.

    Counting separators independently matters: a file full of '/' still
    parses into a haplotype-shaped array, so a parser that reports success
    would mark an unphased file as phased and let gamete simulation run on
    arbitrary phase.
    """
    op = gzip.open if str(vcf_path).endswith(".gz") else open
    seen = phased = 0
    with op(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")[9:]
            for g in fields:
                g = g.split(":")[0]
                if "|" in g:
                    seen += 1
                    phased += 1
                elif "/" in g:
                    seen += 1
            if seen >= max_rows * 10:
                break
    return (phased / seen) if seen else float("nan")


def read_sex(path):
    df = pd.read_csv(path)
    idc = next((c for c in df.columns
                if c.lower() in ("id", "iid", "sample", "individual")),
               df.columns[0])
    sxc = next((c for c in df.columns
                if c.lower() in ("sex", "gender", "\u6027\u522b")), df.columns[-1])
    # Chinese tokens are escaped, not removed: real input files use them.
    m = {"m": "M", "male": "M", "1": "M", "\u516c": "M",
         "f": "F", "female": "F", "2": "F", "\u6bcd": "F"}
    vals = df[sxc].astype(str).str.strip().str.lower().map(
        lambda v: m.get(v, "unknown"))
    return dict(zip(df[idc].astype(str), vals))


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def kernel_diagnostics(name, K):
    n = K.shape[0]
    d = np.diag(K)
    off = K[~np.eye(n, dtype=bool)]
    mean_diag = float(np.mean(d))
    ratio = float(np.mean(np.abs(off)) / max(abs(mean_diag), 1e-12))
    I = np.eye(n)
    cor_I = float(np.corrcoef(K.ravel(), I.ravel())[0, 1])
    ev = np.linalg.eigvalsh(K)
    ev = np.clip(ev, 0, None)
    tot = ev.sum()
    # participation ratio: (sum L)^2 / sum L^2. Equals n for the identity and
    # falls towards 1 as variance concentrates on one direction.
    pr = float(tot ** 2 / max(np.sum(ev ** 2), 1e-12))
    # number of leading eigenvalues holding 90 % of the trace
    cum = np.cumsum(np.sort(ev)[::-1]) / max(tot, 1e-12)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    degenerate = (ratio < DEGENERATE_RATIO) and (cor_I > DEGENERATE_COR_I)
    return dict(kernel=name, n=n, mean_diag=mean_diag, off_diag_ratio=ratio,
                cor_with_I=cor_I, participation_ratio=pr,
                n_eigen_for_90pct=n90, top_eigenvalue=float(ev.max()),
                degenerate=bool(degenerate))


def kernel_collinearity(K):
    names = list(K)
    n = K[names[0]].shape[0]
    mask = ~np.eye(n, dtype=bool)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = float(np.corrcoef(K[a][mask], K[b][mask])[0, 1])
            rows.append(dict(kernel_1=a, kernel_2=b, off_diag_correlation=r))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", required=True,
                    help="PLINK prefix of the reference population")
    ap.add_argument("--candidates", default=None,
                    help="PLINK prefix of an independent candidate population")
    ap.add_argument("--sex", default=None, help="CSV with an id and a sex column")
    ap.add_argument("--vcf", default=None,
                    help="VCF of the same population, to test phase")
    ap.add_argument("--cache_dir", default="adept_kernel_cache")
    ap.add_argument("--skip_kernels", action="store_true",
                    help="capability detection only; skip kernel construction")
    ap.add_argument("--out", default="capability_audit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    say("=" * 74)
    say("CAPABILITY DETECTION")
    say("=" * 74)
    X, chrom, snp, ids, sex_fam = load_plink(a.bfile)
    n, m = X.shape
    say(f"  {n} individuals x {m} markers")

    het_per_ind = np.mean(X == 1, axis=1)
    het = float(het_per_ind.mean())
    Ha, Hd = P.noia(X)
    hd_var = np.var(Hd, axis=0)
    n_no_het = int(np.sum(np.sum(X == 1, axis=0) == 0))
    say(f"  mean heterozygosity per individual : {het:.6f}")
    say(f"  markers with no heterozygote at all: {n_no_het} of {m} "
        f"({100 * n_no_het / m:.1f} %)")
    say(f"  median per-marker dominance-code variance: "
        f"{float(np.median(hd_var)):.3e}")

    cap = {}
    cap["dominance"] = ("blocked" if het <= HET_BLOCK_THRESHOLD
                        else "available")
    cap["AD_and_DD_families"] = cap["dominance"]
    # On genotypes scored 0 and 2 the exclusive-or coding equals the additive
    # coding shifted by a constant, so conditional on both main effects it is
    # the multiplicative family written twice.
    cap["XOR_coding"] = ("blocked (linear transform of the additive coding "
                         "on 0/2 genotypes)" if het <= 1e-4 else "available")
    cap["epistasis_MULT"] = "available"

    if a.vcf:
        pf = phased_fraction(a.vcf)
        say(f"  phased genotype fraction in {os.path.basename(a.vcf)}: "
            f"{pf:.4f}")
        cap["gamete_simulation"] = ("available" if pf >= 0.99
                                    else "blocked (genotypes are not phased)")
    else:
        cap["gamete_simulation"] = "blocked (no phased file supplied)"

    if a.sex:
        sx = read_sex(a.sex)
        nm = sum(1 for v in sx.values() if v == "M")
        nf = sum(1 for v in sx.values() if v == "F")
        say(f"  recorded sex: {nm} males, {nf} females")
        cap["mating_design"] = f"male x female ({nm} x {nf})"
    else:
        coded = int(np.isin(sex_fam, ["1", "2"]).sum())
        if coded >= 0.9 * n:
            nm = int(np.sum(sex_fam == "1"))
            nf = int(np.sum(sex_fam == "2"))
            say(f"  recorded sex: taken from the .fam, {nm} males, "
                f"{nf} females")
            cap["mating_design"] = f"male x female ({nm} x {nf}), from .fam"
        else:
            say(f"  recorded sex: absent (only {coded} of {n} individuals "
                f"are coded 1 or 2 in the .fam)")
            cap["mating_design"] = ("all pairwise combinations; a departure "
                                    "from a biological mating design")

    if a.candidates:
        Xc_, chrom_c, snp_c, ids_c, _ = load_plink(a.candidates)
        if len(snp_c) != m or not np.array_equal(snp_c, snp):
            raise SystemExit(
                "Marker sets differ between the reference and candidate "
                "populations. This is enforced as an assertion, not a "
                "warning: a matrix product across mismatched loci still "
                "returns numbers.")
        say(f"  independent candidates: {len(ids_c)} individuals, marker set "
            f"identical")
        cap["allocation_scope"] = f"independent candidate population (n={len(ids_c)})"
        p_ref = X.mean(0) / 2
        p_can = Xc_.mean(0) / 2
        r_af = float(np.corrcoef(p_ref, p_can)[0, 1])
        say(f"  allele-frequency correlation reference vs candidates: "
            f"{r_af:.4f}" + ("   <- below 0.95: allocation extrapolates"
                             if r_af < 0.95 else ""))
        cap["allele_freq_correlation"] = r_af
    else:
        cap["allocation_scope"] = ("within the reference population "
                                   "(no independent candidate set)")

    say("\n  capability profile:")
    for k, v in cap.items():
        say(f"    {k:26s} {v}")

    diag_rows = []
    coll = None
    if not a.skip_kernels:
        say("\n" + "=" * 74)
        say("KERNEL DEGENERACY AUDIT")
        say("=" * 74)
        qc = Q.CFG()
        qc.CACHE_DIR = a.cache_dir
        qc.BFILE = a.bfile
        qc.DO_CIS = False
        K, meta = Q.build_kernels(qc, X, Ha, Hd, chrom, keep_xc=False)
        if cap["dominance"].startswith("blocked"):
            say("  note: the dominance kernel and the dominance-derived "
                "families are audited below for completeness but are BLOCKED "
                "for this material and take no part in any decision.")
        for name in K:
            diag_rows.append(kernel_diagnostics(name, K[name]))
        D = pd.DataFrame(diag_rows)
        say("\n" + D.round(5).to_string(index=False))
        say("\n  off_diag_ratio       mean |off-diagonal| / mean diagonal")
        say("  cor_with_I           correlation of the kernel with the "
            "identity matrix")
        say("  participation_ratio  (sum L)^2 / sum L^2, which equals n for "
            "the identity")
        say("  n_eigen_for_90pct    leading eigenvalues holding 90 % of the "
            "trace")
        say(f"  a kernel is flagged degenerate when off_diag_ratio < "
            f"{DEGENERATE_RATIO} and cor_with_I > {DEGENERATE_COR_I}")
        deg = D[D.degenerate]
        if len(deg):
            say("\n  DEGENERATE, exclude before any decision: "
                + ", ".join(deg.kernel))
            say("  Fitting one of these as a random effect is a ridge "
                "penalty. It can raise held-out accuracy across splits and "
                "pass a positive-fraction criterion while describing no "
                "covariance between individuals.")
        else:
            say("\n  no kernel met the degeneracy criterion")
        D.to_csv(os.path.join(a.out, "kernel_degeneracy_audit.csv"),
                 index=False)

        coll = kernel_collinearity(K)
        coll.to_csv(os.path.join(a.out, "kernel_collinearity.csv"),
                    index=False)
        say("\n" + "=" * 74)
        say("KERNEL COLLINEARITY (off-diagonal correlations)")
        say("=" * 74)
        say(coll.round(4).to_string(index=False))
        hi = coll[coll.off_diag_correlation.abs() > 0.5]
        if len(hi):
            say("\n  |r| > 0.5 for: "
                + ", ".join(f"{r.kernel_1}~{r.kernel_2} "
                            f"({r.off_diag_correlation:.3f})"
                            for r in hi.itertuples()))
            say("  Variance components are not separately identified at these "
                "correlations. Report out-of-sample prediction gain; treat "
                "variance fractions as descriptive only.")

    prof = dict(bfile=a.bfile, n_individuals=int(n), n_markers=int(m),
                mean_heterozygosity=het,
                markers_without_heterozygote=n_no_het,
                median_dominance_code_variance=float(np.median(hd_var)),
                capability=cap,
                het_block_threshold=HET_BLOCK_THRESHOLD,
                degenerate_criterion=dict(off_diag_ratio_below=DEGENERATE_RATIO,
                                          cor_with_I_above=DEGENERATE_COR_I))
    with open(os.path.join(a.out, "capability_profile.json"), "w") as f:
        json.dump(prof, f, indent=2)
    say(f"\n-> {a.out}/capability_profile.json"
        + ("" if a.skip_kernels else
           f"\n-> {a.out}/kernel_degeneracy_audit.csv"
           f"\n-> {a.out}/kernel_collinearity.csv"))


if __name__ == "__main__":
    main()
