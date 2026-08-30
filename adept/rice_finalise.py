#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rice_finalise.py
================

The three things rice still needs before it can go into the paper, in one run.

  [1] STRUCTURE   How much of the additive accuracy is subpopulation membership
                  rather than within-subpopulation genomic prediction. RDP1 is
                  strongly stratified (indica / japonica / aus / aromatic) and
                  the pipeline fits an intercept only -- pc.N_PCS = 0 is set in
                  pig_epistasis_pipeline.run(), so the log line claiming
                  "intercept + structure PCs" is wrong and no PC ever enters the
                  design matrix. This stage quantifies the consequence instead
                  of arguing about it.

  [2] DKERNELS    Why A+AD_t / A+DD_t / A+XOR_t change the accuracy and halve
                  its standard deviation in a population with 0.036 percent
                  heterozygosity, where the dominance coding is near-degenerate
                  and those kernels should carry no information. The hypothesis
                  to test is that they are proportional to the identity matrix,
                  in which case adding one as a random effect is algebraically a
                  ridge penalty: it stabilises predictions across splits without
                  explaining any genetic variance. If that holds, Blast
                  resistance passing the 0.80 criterion on A+DD_t is an artefact
                  of the criterion, not a finding.

  [3] CALIBRATE   Injection calibration of the epistasis arm, matched to the
                  screening configuration actually used (seed 2024,
                  test_frac 0.20, 20 splits, MULT_t, threshold 0.80). Without
                  it, "no epistasis in rice" can only be written as "did not
                  meet the criterion", never as "the effect is absent" -- the
                  same rule already applied to the mussel.

USAGE
-----
    python rice_finalise.py --selftest
    python rice_finalise.py \
        --bfile rice_imputed_final_pruned \
        --traits "Plant height" "Alkali spreading value" "Amylose content" \
                 "Protein content" "Blast resistance" \
        --out rice_final

Stage 3 is the long one. It checkpoints after every (level, replicate) cell to
_calib/calibration_raw.csv and skips finished cells on restart, so it can be
interrupted and resumed. Run stages selectively with --stages if needed:

    python rice_finalise.py ... --stages structure,dkernels     # minutes
    python rice_finalise.py ... --stages calibrate              # hours

COST
----
One REML fit at n~360 takes 2-3 s in this environment. Stage 3 costs
levels x reps x splits x 2 fits; the default grid (5 x 20 x 20 x 2 = 4000 fits)
is roughly 3 hours. Halve --calib_reps if that is too long; detection rate then
resolves to 0.10 instead of 0.05.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def say(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# Simulation and the screening rule live in power_curve.py, shared with the
# agent and with every other material, so a calibration curve computed here is
# produced by exactly the same code that produces the pig and mussel curves.
# --------------------------------------------------------------------------- #
from power_curve import (sample_from_kernel, simulate_phenotype, verdict,
                         identity_likeness)


# --------------------------------------------------------------------------- #
# Pipeline-backed stages                                                       #
# --------------------------------------------------------------------------- #

def load_everything(bfile: str, first_trait: str, out_dir: str, cache_dir: str):
    """Load genotypes once and build every kernel once, exactly as the pipeline does."""
    import pig_epik_pipeline as P
    import pig_epistasis_pipeline as Q

    Q.install_reml_speedup()

    pc = P.CFG()
    pc.BFILE = bfile
    pc.N_PCS = 0
    pc.PHENO_FILE = f"{first_trait}.phe"
    pc.TRAIT = first_trait
    pc.COV_FILE = f"{first_trait}.cov" if os.path.exists(f"{first_trait}.cov") else None
    pc.QCOV_FILE = f"{first_trait}.qcov" if os.path.exists(f"{first_trait}.qcov") else None
    X, _y0, chrom, pos, snp_names, _extra = P.load_real(pc)
    n, m = X.shape
    say(f"[data] {n} individuals x {m} SNPs")

    Ha, Hd = P.noia(X)

    kcfg = Q.CFG()
    kcfg.BFILE = bfile
    kcfg.CACHE_DIR = cache_dir
    kcfg.DO_CIS = False
    kcfg.ADD_HAD = True
    kcfg.EXACT_TYPES = ("MULT", "AD", "DD", "XOR")
    K, meta = Q.build_kernels(kcfg, X, Ha, Hd, chrom, keep_xc=False)

    labels, how = P.structure_clusters(K["A"], None, 10, 2024)
    say(f"[structure] clustering: {how}")
    return dict(P=P, Q=Q, pc=pc, X=X, Ha=Ha, Hd=Hd, chrom=chrom, K=K, meta=meta,
                labels=labels, n=n, m=m)


def load_trait(env, trait: str) -> Optional[np.ndarray]:
    """Phenotype vector for one trait, or None if its individual set differs."""
    P, pc = env["P"], env["pc"]
    f = f"{trait}.phe"
    if not os.path.exists(f):
        say(f"  [skip] {trait}: {f} not found")
        return None
    pc.PHENO_FILE, pc.TRAIT = f, trait
    pc.COV_FILE = f"{trait}.cov" if os.path.exists(f"{trait}.cov") else None
    pc.QCOV_FILE = f"{trait}.qcov" if os.path.exists(f"{trait}.qcov") else None
    Xt, y, _, _, _, _ = P.load_real(pc)
    if Xt.shape[0] != env["n"]:
        say(f"  [skip] {trait}: {Xt.shape[0]} individuals vs {env['n']} in the "
            f"kernel set. Traits with different missingness cannot share "
            f"pre-built kernels; run this trait separately.")
        return None
    return y


# ---- [1] structure ---------------------------------------------------------- #

def stage_structure(env, traits: List[str], out_dir: str, n_pc: int = 10
                    ) -> pd.DataFrame:
    """
    Split the additive accuracy into what subpopulation membership alone gives
    and what the genomic model adds on top.

    Three numbers per trait:
      r2_pc          variance in the phenotype explained by the leading PCs
      acc_pc_only    accuracy of a model whose only predictor is those PCs
      acc_additive   accuracy of the additive genomic model, as the paper reports
    A large acc_pc_only relative to acc_additive means the headline accuracy is
    largely between-subpopulation prediction. That does not invalidate it -- it
    is what genomic selection on a diversity panel does -- but it must be stated,
    because a reader will otherwise compare 0.76 in rice with 0.38 in pig as if
    they measured the same thing.
    """
    Q = env["Q"]
    K, labels, n = env["K"], env["labels"], env["n"]

    w, V = np.linalg.eigh((K["A"] + K["A"].T) / 2.0)
    order = np.argsort(-w)
    PC = V[:, order[:n_pc]] * np.sqrt(np.maximum(w[order[:n_pc]], 0.0))
    pc_var_share = float(np.sum(np.maximum(w[order[:n_pc]], 0)) /
                         max(np.sum(np.maximum(w, 0)), 1e-12))
    say(f"  leading {n_pc} eigenvectors carry {pc_var_share:.3f} of the "
        f"additive kernel's total variance")

    rows = []
    for trait in traits:
        y = load_trait(env, trait)
        if y is None:
            continue
        Xfix = np.ones((n, 1))

        # phenotype variance explained by the PCs (in-sample R^2)
        A_ = np.column_stack([np.ones(n), PC])
        beta, *_ = np.linalg.lstsq(A_, y, rcond=None)
        resid = y - A_ @ beta
        r2_pc = float(1.0 - resid.var() / max(y.var(), 1e-12))

        accs_pc, accs_add = [], []
        for si in range(20):
            seed = 2024 + si * 101
            tr, te = Q.make_split(n, labels, "random", 0.20, seed)
            # PC-only predictor, fitted on the training split
            b, *_ = np.linalg.lstsq(A_[tr], y[tr], rcond=None)
            accs_pc.append(Q.pcc(y[te] - y[tr].mean(), A_[te] @ b))
            gb, ya, _ = Q.fit_gebv(y, Xfix, K, ["A"], tr, te)
            accs_add.append(Q.pcc(ya, gb) if gb is not None else np.nan)

        rows.append(dict(
            trait=trait, n=n, r2_pc=round(r2_pc, 4),
            acc_pc_only=round(float(np.nanmean(accs_pc)), 4),
            acc_additive=round(float(np.nanmean(accs_add)), 4),
            acc_pc_share=round(float(np.nanmean(accs_pc) /
                                     max(np.nanmean(accs_add), 1e-12)), 4)))
        say(f"    {trait:26s} R2(PC)={rows[-1]['r2_pc']:.3f}  "
            f"acc(PC only)={rows[-1]['acc_pc_only']:.3f}  "
            f"acc(additive)={rows[-1]['acc_additive']:.3f}")

    df = pd.DataFrame(rows)
    df.attrs["pc_var_share"] = pc_var_share
    p = os.path.join(out_dir, "rice_structure_attribution.csv")
    df.to_csv(p, index=False)
    say(f"  -> {p}")
    return df


# ---- [2] dominance-derived kernels ------------------------------------------ #

def stage_dkernels(env, out_dir: str) -> pd.DataFrame:
    """
    Is the dominance coding degenerate, and are the D-derived kernels acting as
    a ridge penalty rather than as genetic covariance?
    """
    X, Hd, K = env["X"], env["Hd"], env["K"]
    n, m = X.shape

    het_per_ind = np.mean(X == 1, axis=1)
    het_per_snp = np.mean(X == 1, axis=0)
    say(f"  heterozygosity: mean per individual {het_per_ind.mean():.6f}, "
        f"max {het_per_ind.max():.6f}")
    say(f"  markers with no heterozygote at all: "
        f"{int(np.sum(het_per_snp == 0))}/{m} "
        f"({np.mean(het_per_snp == 0):.3f})")
    say(f"  var(Hd) per marker: median {np.median(np.var(Hd, axis=0)):.3e}, "
        f"max {np.max(np.var(Hd, axis=0)):.3e}")

    rows = []
    for name in ("A", "D", "MULT_t", "AD_t", "DD_t", "XOR_t", "HAD"):
        if name not in K:
            continue
        s = identity_likeness(K[name])
        s["kernel"] = name
        s["uses_dominance_coding"] = name in ("D", "AD_t", "DD_t")
        rows.append(s)
        say(f"    {name:8s} offdiag/diag={s['offdiag_over_diag']:.4f}  "
            f"corr_with_I={s['corr_with_identity']:+.4f}  "
            f"effective_rank={s['effective_rank']:.1f}")

    df = pd.DataFrame(rows)[["kernel", "uses_dominance_coding",
                             "offdiag_over_diag", "corr_with_identity",
                             "effective_rank", "top_eigen_share", "n"]]
    p = os.path.join(out_dir, "rice_kernel_degeneracy.csv")
    df.to_csv(p, index=False)

    meta = dict(mean_heterozygosity=float(het_per_ind.mean()),
                max_heterozygosity=float(het_per_ind.max()),
                n_snps=int(m), n_snps_no_het=int(np.sum(het_per_snp == 0)),
                median_var_Hd=float(np.median(np.var(Hd, axis=0))))
    json.dump(meta, open(os.path.join(out_dir, "rice_heterozygosity.json"), "w"),
              indent=2)

    say(f"  -> {p}")
    say("\n  HOW TO READ THIS: a kernel with offdiag/diag near 0 and "
        "corr_with_identity near 1 is\n"
        "  proportional to the identity. Adding it as a random effect is a "
        "ridge penalty --\n"
        "  it stabilises the accuracy across splits without explaining "
        "covariance between\n"
        "  individuals. If AD_t / DD_t look like that while MULT_t does not, "
        "the Blast\n"
        "  resistance pass on A+DD_t is an artefact of the criterion and those "
        "kernels\n"
        "  should be blocked by capability detection alongside the dominance "
        "module.")
    return df


# ---- [3] calibration -------------------------------------------------------- #

def stage_calibrate(env, out_dir: str, levels: List[float], reps: int,
                    n_splits: int, h2_sim: float, threshold: float,
                    epi_kernel: str) -> pd.DataFrame:
    """
    Injection calibration for the epistasis arm, matched to the screening
    configuration. The dominance arm is not calibrated because the dominance
    module is blocked by capability detection in this material -- calibrating a
    module that will never run would be reporting a number nobody can use.
    """
    Q = env["Q"]
    K, labels, n = env["K"], env["labels"], env["n"]
    if epi_kernel not in K:
        raise SystemExit(f"epistatic kernel {epi_kernel!r} not built; have "
                         f"{sorted(K)}")

    ck_dir = os.path.join(out_dir, "_calib")
    os.makedirs(ck_dir, exist_ok=True)
    raw_path = os.path.join(ck_dir, "calibration_raw.csv")
    done: Dict[Tuple[float, int], bool] = {}
    rows: List[dict] = []
    if os.path.exists(raw_path):
        prev = pd.read_csv(raw_path)
        rows = prev.to_dict("records")
        done = {(round(float(r["injected"]), 6), int(r["rep"])): True
                for r in rows}
        say(f"  resuming: {len(rows)} cell(s) already done")

    Xfix = np.ones((n, 1))
    t0 = time.time()
    total = len(levels) * reps
    for lvl in levels:
        for rep in range(reps):
            if (round(float(lvl), 6), rep) in done:
                continue
            rng = np.random.default_rng(9000 + rep)      # same draw at every
            y = simulate_phenotype(K["A"], K[epi_kernel],  # level, so the curve
                                   h2_sim, float(lvl), rng)  # is paired
            deltas = []
            for si in range(n_splits):
                seed = 2024 + si * 101
                tr, te = Q.make_split(n, labels, "random", 0.20, seed)
                g0, ya, _ = Q.fit_gebv(y, Xfix, K, ["A"], tr, te)
                g1, _, _ = Q.fit_gebv(y, Xfix, K, ["A", epi_kernel], tr, te)
                if g0 is None or g1 is None:
                    continue
                deltas.append(Q.pcc(ya, g1) - Q.pcc(ya, g0))
            mean_d, pos, hit = verdict(np.array(deltas), threshold)
            rows.append(dict(module="epistasis", injected=float(lvl), rep=rep,
                             n_splits=len(deltas), dPCC=mean_d,
                             pos_frac=pos, detected=int(hit)))
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            k = len(rows)
            say(f"    injected={lvl:.2f} rep {rep + 1}/{reps}  "
                f"dPCC={mean_d:+.5f} pos_frac={pos:.2f} "
                f"{'HIT' if hit else '.'}   "
                f"[{k}/{total}, {(time.time() - t0) / 60:.1f} min elapsed]")

    R = pd.DataFrame(rows)
    summ = (R.groupby("injected")
              .agg(n_reps=("rep", "count"), mean_dPCC=("dPCC", "mean"),
                   median_dPCC=("dPCC", "median"),
                   detection_rate=("detected", "mean"))
              .reset_index())
    p = os.path.join(out_dir, "rice_calibration.csv")
    summ.to_csv(p, index=False)
    say("\n" + summ.round(5).to_string(index=False))
    say(f"  -> {p}")

    fp = summ.loc[summ.injected == 0.0, "detection_rate"]
    if len(fp):
        say(f"\n  false-positive rate at zero injection: {float(fp.iloc[0]):.2f}")
    ok = summ[(summ.injected > 0) & (summ.detection_rate >= 0.80)]
    if len(ok):
        say(f"  detection limit (80% power): {float(ok.injected.iloc[0]):.2f} "
            f"of phenotypic variance")
    else:
        say("  NO injected level reached 80% power. The epistasis arm has no "
            "usable detection\n  limit in this material, so its negatives "
            "carry no information and must be\n  reported as insufficient "
            "power, not as absence of effect.")
    return summ


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bfile", default="rice_imputed_final_pruned")
    ap.add_argument("--traits", nargs="+", default=[
        "Plant height", "Alkali spreading value", "Amylose content",
        "Protein content", "Blast resistance"])
    ap.add_argument("--out", default="rice_final")
    ap.add_argument("--cache", default="rice_kernel_cache")
    ap.add_argument("--stages", default="structure,dkernels,calibrate")
    ap.add_argument("--levels", nargs="*", type=float,
                    default=[0.0, 0.05, 0.15, 0.30, 0.50])
    ap.add_argument("--calib_reps", type=int, default=20)
    ap.add_argument("--calib_splits", type=int, default=20)
    ap.add_argument("--h2_sim", type=float, default=0.60,
                    help="total simulated heritability. Rice additive variance "
                         "fractions came out at 0.60-0.89, so 0.30 (the pig "
                         "value) would understate this material")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--epi_kernel", default="MULT_t")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(_selftest())

    stages = {s.strip() for s in a.stages.split(",") if s.strip()}
    os.makedirs(a.out, exist_ok=True)

    say("=" * 78)
    say("rice_finalise -- structure attribution, kernel degeneracy, calibration")
    say("=" * 78)
    env = load_everything(a.bfile, a.traits[0], a.out, a.cache)

    if "structure" in stages:
        say("\n[1] STRUCTURE: how much of the additive accuracy is subpopulation")
        stage_structure(env, a.traits, a.out)

    if "dkernels" in stages:
        say("\n[2] DKERNELS: is the dominance coding degenerate, and are the")
        say("    D-derived kernels acting as a ridge penalty?")
        stage_dkernels(env, a.out)

    if "calibrate" in stages:
        say(f"\n[3] CALIBRATE: epistasis arm, {len(a.levels)} levels x "
            f"{a.calib_reps} reps x {a.calib_splits} splits")
        say(f"    h2_sim={a.h2_sim}  kernel={a.epi_kernel}  "
            f"threshold={a.threshold}")
        stage_calibrate(env, a.out, a.levels, a.calib_reps, a.calib_splits,
                        a.h2_sim, a.threshold, a.epi_kernel)

    say(f"\nAll requested stages done. Results in {a.out}/")


# --------------------------------------------------------------------------- #

def _selftest() -> int:
    ok = True
    rng = np.random.default_rng(0)
    n = 120
    Z = rng.standard_normal((n, 200))
    Ka = Z @ Z.T / 200
    Ka /= np.mean(np.diag(Ka))
    W = rng.standard_normal((n, 150))
    Ke = W @ W.T / 150
    Ke /= np.mean(np.diag(Ke))

    # (1) sampling reproduces the kernel's covariance structure
    # 2000 draws for a 120x120 covariance; at 600 the Monte-Carlo error alone
    # holds the correlation near 0.86, which is sampling noise, not a bug
    S = np.array([sample_from_kernel(Ka, np.random.default_rng(i))
                  for i in range(2000)])
    emp = np.cov(S.T)
    r = np.corrcoef(emp[np.triu_indices(n, 1)], Ka[np.triu_indices(n, 1)])[0, 1]
    if r < 0.93:
        print(f"FAIL: sampled covariance correlates {r:.3f} with the kernel")
        ok = False

    # (2) the injected variance really lands in the phenotype
    y0 = simulate_phenotype(Ka, Ke, 0.6, 0.0, np.random.default_rng(1))
    y5 = simulate_phenotype(Ka, Ke, 0.6, 0.5, np.random.default_rng(1))
    ge = sample_from_kernel(Ke, np.random.default_rng(1))  # not the same draw
    if abs(y0.var() - y5.var()) > 0.6 * max(y0.var(), 1.0):
        print("FAIL: injection changed the total variance more than expected")
        ok = False
    # a heavier injection must raise the correlation with an epistatic direction
    rng2 = np.random.default_rng(4)
    lo, hi = [], []
    for i in range(40):
        r_ = np.random.default_rng(100 + i)
        g = sample_from_kernel(Ke, np.random.default_rng(100 + i))
        yl = simulate_phenotype(Ka, Ke, 0.6, 0.0, np.random.default_rng(100 + i))
        yh = simulate_phenotype(Ka, Ke, 0.6, 0.5, np.random.default_rng(100 + i))
        lo.append(abs(np.corrcoef(yl, g)[0, 1]))
        hi.append(abs(np.corrcoef(yh, g)[0, 1]))
    if np.mean(hi) <= np.mean(lo):
        print(f"FAIL: injection does not increase alignment with the epistatic "
              f"component ({np.mean(lo):.3f} -> {np.mean(hi):.3f})")
        ok = False

    # (3) rejecting an injection larger than h2
    try:
        simulate_phenotype(Ka, Ke, 0.3, 0.5, rng)
        print("FAIL: injection above h2 was accepted")
        ok = False
    except ValueError:
        pass

    # (4) the screening rule
    m_, p_, h_ = verdict(np.array([0.01] * 17 + [-0.01] * 3), 0.80)
    if not h_ or abs(p_ - 0.85) > 1e-9:
        print(f"FAIL: verdict 17/20 positive -> pos={p_}, hit={h_}"); ok = False
    m_, p_, h_ = verdict(np.array([0.01] * 15 + [-0.01] * 5), 0.80)
    if h_:
        print("FAIL: 15/20 positive should not fire at threshold 0.80"); ok = False
    m_, p_, h_ = verdict(np.array([-0.01] * 20), 0.80)
    if h_:
        print("FAIL: all-negative fired"); ok = False

    # (5) identity likeness separates a ridge-like kernel from a genetic one
    idl_I = identity_likeness(np.eye(n))
    idl_K = identity_likeness(Ka)
    if not (idl_I["corr_with_identity"] > 0.99):
        print("FAIL: identity not recognised"); ok = False
    if not (idl_I["offdiag_over_diag"] < 1e-9):
        print("FAIL: identity offdiag ratio wrong"); ok = False
    if not (idl_K["corr_with_identity"] < idl_I["corr_with_identity"]):
        print("FAIL: genetic kernel scored as identity-like"); ok = False
    if not (idl_K["effective_rank"] > 10 * 1.0):
        print(f"FAIL: effective rank {idl_K['effective_rank']:.1f} implausible")
        ok = False
    # a near-degenerate kernel (all individuals almost identical) must look
    # nothing like the identity either -- the two failure modes are distinct
    Kflat = np.ones((n, n)) + 1e-6 * np.eye(n)
    idl_F = identity_likeness(Kflat)
    if idl_F["effective_rank"] > 1.5:
        print(f"FAIL: rank-1 kernel got effective rank "
              f"{idl_F['effective_rank']:.2f}"); ok = False

    print("SELFTEST: " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    main()
