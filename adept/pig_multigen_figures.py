#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Multi-generation mate-allocation simulation and publication figures
 ------------------------------------------------------------------------------
 Companion module to pig_epistasis_pipeline.py. Adds the two things the main
 pipeline was missing for a manuscript:

   [E] MULTIGEN   Five generations of mate allocation under five strategies,
                  reporting realised genetic gain, genetic SD and inbreeding
                  per generation. Single-generation overlap statistics only
                  show that different crosses are recommended; they do NOT
                  show whether that translates into more gain after several
                  generations. This stage answers that.

   [F] FIGURES    Publication figures matching the manuscript structure.

================================================================================
 [E] SIMULATION DESIGN
================================================================================
 Strategies compared (identical in every respect except the ranking criterion,
 so that ranking is the only variable):
     RANDOM      crosses drawn at random                      (negative control)
     ADD         mid-parent additive value
     DOM         expected offspring dominance value only
     ADD+DOM     additive + dominance
     ADD+DOM+EPI additive + dominance + epistasis

 Per generation:
   1. evaluate every pairwise cross among the current animals
   2. rank by the strategy criterion, allocate greedily under the inbreeding
      ceiling and the maximum-use-per-animal constraint
   3. simulate N_SCREEN gametes per parent per cross, form offspring, keep the
      best OFF_PER_CROSS by realised genetic value
   4. the retained offspring become the next generation

 Step 3 matters: without it the simulation credits the ranking criterion for a
 within-cross selection step it never performed, and understates the realised
 gain of every strategy equally.

 TWO LIMITATIONS THAT MUST BE STATED IN THE MANUSCRIPT
 -----------------------------------------------------
 1. Effects are taken as known. Marker effects and variance components are
    estimated once on the reference population and then treated as true. The
    simulation therefore measures how much gain a given ALLOCATION STRATEGY
    can realise given those estimates; it cannot validate the estimates
    themselves. This is the standard convention in mate-allocation simulation.
 2. Free recombination. The genotypes are unphased, so haplotypes cannot be
    reconstructed and gametes are simulated by independent transmission at
    each locus. This is internally consistent with the trans-only (unlinked)
    epistatic kernel used throughout, but it ignores within-chromosome linkage
    and therefore does not reproduce linkage-driven Mendelian sampling
    covariance. Gains are consequently a lower bound with respect to
    linkage-based segregation variance.

================================================================================
 USAGE
     python pig_multigen_figures.py --selftest
     python pig_multigen_figures.py --trait bf --stage multigen --generations 5
     python pig_multigen_figures.py --stage figures --results_glob "pig_results_*"
================================================================================
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pig_epik_pipeline as P
import pig_epistasis_pipeline as Q

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

STRATEGIES = ("RANDOM", "ADD", "DOM", "ADD+DOM", "ADD+DOM+EPI")
PALETTE = {"RANDOM": "#9e9e9e", "ADD": "#1f77b4", "DOM": "#2ca02c",
           "ADD+DOM": "#ff7f0e", "ADD+DOM+EPI": "#d62728"}


class MCFG:
    BFILE        = "genotypes"      # set via --bfile
    TRAIT        = "bf"
    OUT_DIR      = "pig_multigen_out"
    SEED         = 2024
    N_CANDIDATES = 300
    N_CROSSES    = 60
    OFF_PER_CROSS = 5
    N_SCREEN     = 50      # gametes screened per parent per cross
    MAX_USE      = 5
    F_MAX        = 0.0625
    GENERATIONS  = 5
    N_REPS       = 3       # independent simulation replicates


def say(*a):
    print(*a, flush=True)


# ============================================================================
# Genetic-value machinery
# ============================================================================
class GeneticModel:
    """
    Holds everything needed to score an arbitrary genotype matrix:
    back-solved additive and dominance marker effects, plus the reference-based
    machinery for the trans epistatic term.
    """

    def __init__(self, X, Ha, Hd, chrom, K, meta, y, Xfix):
        n, m = X.shape
        self.chrom = chrom
        self.mu = X.mean(0)
        self.p_het = np.mean(X == 1, axis=0)
        var_A = np.var(Ha, axis=0)
        self.beta_d = np.where(var_A > 1e-12,
                               (self.p_het * (1 - self.mu)) / np.maximum(var_A, 1e-12), 0.0)
        self.Ha_ref = Ha

        keys = [k for k in ("A", "D", "MULT_t") if k in K]
        Ktt = [K[k] for k in keys]
        mm = P.REMLgrad().fit(y, Xfix, Ktt)
        v = np.asarray(mm.variances_, float)
        V = sum(s * A for s, A in zip(v[:-1], Ktt)) + (v[-1] + 1e-8) * np.eye(n)
        c = cho_factor(V, lower=True)
        b = np.linalg.solve(Xfix.T @ cho_solve(c, Xfix), Xfix.T @ cho_solve(c, y))
        self.alpha = cho_solve(c, y - Xfix @ b)
        self.var = {k: float(s) for k, s in zip(keys, v[:-1])}
        self.var["_resid"] = float(v[-1])
        tot = sum(self.var.values())
        self.prop = {k: self.var[k] / tot for k in self.var}

        p = self.mu / 2
        cA = 2 * np.sum(p * (1 - p)) + 1e-12
        self.a_hat = self.var.get("A", 0.0) * (Ha.T @ self.alpha) / (meta["sA"] * cA)
        cD = np.sum(np.var(Hd, 0)) + 1e-12
        self.d_hat = self.var.get("D", 0.0) * (Hd.T @ self.alpha) / (meta["sD"] * cD)

        self.use_epi = "MULT_t" in self.var and self.var["MULT_t"] > 0
        if self.use_epi:
            probe = np.arange(min(50, n))
            Gp = Ha[probe] @ Ha.T
            raw = Gp[np.arange(len(probe)), probe] ** 2
            for ch in np.unique(chrom):
                idx = chrom == ch
                Gc = Ha[probe][:, idx] @ Ha[:, idx].T
                raw = raw - Gc[np.arange(len(probe)), probe] ** 2
            s_AA = float(np.mean(raw / np.maximum(np.diag(K["MULT_t"])[probe], 1e-12)))
            self.scale_AA = self.var["MULT_t"] / max(s_AA, 1e-12)
        self.KA_raw = meta["KA_raw"]

    # ---- genotype -> NOIA codes -------------------------------------------
    def codes(self, G):
        Ha = G - self.mu[None, :]
        H = (G == 1).astype(float)
        Hd = H - self.p_het[None, :] - self.beta_d[None, :] * Ha
        return Ha, Hd

    def expected_codes(self, Gp, Gq):
        """Expected offspring NOIA codes for parents with genotypes Gp, Gq."""
        a, b = Gp / 2.0, Gq / 2.0
        e_ha = (a + b) - self.mu[None, :]
        p_het = a * (1 - b) + (1 - a) * b
        e_hd = p_het - self.p_het[None, :] - self.beta_d[None, :] * e_ha
        return e_ha, e_hd

    # ---- genetic values ----------------------------------------------------
    def epi_value(self, Ha_new, chunk=256):
        """Trans epistatic genetic value for arbitrary individuals."""
        if not self.use_epi:
            return np.zeros(Ha_new.shape[0])
        out = np.empty(Ha_new.shape[0])
        for s in range(0, Ha_new.shape[0], chunk):
            e = min(s + chunk, Ha_new.shape[0])
            t = self.Ha_ref @ Ha_new[s:e].T                  # n_ref x chunk
            acc = t ** 2
            for ch in np.unique(self.chrom):
                idx = self.chrom == ch
                tc = self.Ha_ref[:, idx] @ Ha_new[s:e][:, idx].T
                acc -= tc ** 2
            out[s:e] = self.scale_AA * (self.alpha @ acc)
        return out

    def value(self, G):
        Ha, Hd = self.codes(G)
        return dict(add=Ha @ self.a_hat, dom=Hd @ self.d_hat, epi=self.epi_value(Ha))


# ============================================================================
# [E] multi-generation simulation
# ============================================================================
def sample_gametes(G, n_gam, rng):
    """
    Independent (free-recombination) transmission: each locus contributes an
    allele drawn with probability g/2. Genotypes are unphased, so haplotypes
    cannot be reconstructed and linkage cannot be honoured -- see the module
    docstring for what this implies.
    """
    p = G / 2.0
    return (rng.random((n_gam,) + G.shape) < p[None, :]).astype(np.float64)


def evaluate_crosses(gm, G, pairs_i, pairs_j, chunk=2000):
    """Expected offspring components for every candidate cross."""
    npair = len(pairs_i)
    add = np.empty(npair)
    dom = np.empty(npair)
    epi = np.zeros(npair)
    for s in range(0, npair, chunk):
        e = min(s + chunk, npair)
        e_ha, e_hd = gm.expected_codes(G[pairs_i[s:e]], G[pairs_j[s:e]])
        add[s:e] = e_ha @ gm.a_hat
        dom[s:e] = e_hd @ gm.d_hat
        if gm.use_epi:
            epi[s:e] = gm.epi_value(e_ha)
    return add, dom, epi


def allocate(score, pi, pj, F, cfg):
    """Greedy allocation under inbreeding and maximum-use constraints."""
    order = np.argsort(-score)
    use, chosen = {}, []
    for k in order:
        if F[k] > cfg.F_MAX:
            continue
        a, b = int(pi[k]), int(pj[k])
        if use.get(a, 0) >= cfg.MAX_USE or use.get(b, 0) >= cfg.MAX_USE:
            continue
        use[a] = use.get(a, 0) + 1
        use[b] = use.get(b, 0) + 1
        chosen.append(k)
        if len(chosen) >= cfg.N_CROSSES:
            break
    return np.asarray(chosen, int)


def simulate_multigen(cfg, gm, G0, KA_cand, trait, rep, rng):
    """Run all strategies from the same starting population."""
    rows = []
    for strat in STRATEGIES:
        G = G0.copy()
        r = np.random.default_rng(cfg.SEED + rep * 977 + hash(strat) % 1000)
        KA_cur = KA_cand.copy()
        for gen in range(cfg.GENERATIONS + 1):
            val = gm.value(G)
            tot = val["add"] + val["dom"] + val["epi"]
            rows.append(dict(trait=trait, rep=rep, strategy=strat, generation=gen,
                             mean_total=float(tot.mean()), sd_total=float(tot.std()),
                             mean_add=float(val["add"].mean()),
                             mean_dom=float(val["dom"].mean()),
                             mean_epi=float(val["epi"].mean()),
                             mean_F=float(np.mean(np.diag(KA_cur)) - 1.0),
                             n=len(G)))
            if gen == cfg.GENERATIONS:
                break
            nc = len(G)
            pi, pj = map(np.asarray, zip(*combinations(range(nc), 2)))
            add, dom, epi = evaluate_crosses(gm, G, pi, pj)
            F = KA_cur[pi, pj] / 2.0
            score = {"RANDOM": r.normal(0, 1, len(pi)),
                     "ADD": add, "DOM": dom, "ADD+DOM": add + dom,
                     "ADD+DOM+EPI": add + dom + epi}[strat]
            sel = allocate(score, pi, pj, F, cfg)
            kids = []
            for k in sel:
                a, b = int(pi[k]), int(pj[k])
                ga = sample_gametes(G[a], cfg.N_SCREEN, r)
                gb = sample_gametes(G[b], cfg.N_SCREEN, r)
                cand = ga + gb
                v = gm.value(cand)
                sc = v["add"] + v["dom"] + v["epi"]
                keep = np.argsort(-sc)[:cfg.OFF_PER_CROSS]
                kids.append(cand[keep])
            G = np.vstack(kids)
            Hn = G - gm.mu[None, :]
            p = gm.mu / 2
            cA = 2 * np.sum(p * (1 - p)) + 1e-12
            KA_cur = (Hn @ Hn.T) / cA
    return pd.DataFrame(rows)


def run_multigen(cfg):
    t0 = time.time()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    pc = P.CFG()
    pc.BFILE, pc.N_PCS = cfg.BFILE, 0
    pc.PHENO_FILE, pc.TRAIT = f"{cfg.TRAIT}.phe", cfg.TRAIT
    pc.COV_FILE = f"{cfg.TRAIT}.cov" if os.path.exists(f"{cfg.TRAIT}.cov") else None
    pc.QCOV_FILE = f"{cfg.TRAIT}.qcov" if os.path.exists(f"{cfg.TRAIT}.qcov") else None
    X, y, chrom, pos, snp_names, _ = P.load_real(pc)
    n, m = X.shape
    say(f"[data] {n} animals x {m} SNPs   trait = {cfg.TRAIT}")
    Ha, Hd = P.noia(X)
    qcfg = Q.CFG()
    qcfg.BFILE = cfg.BFILE
    K, meta = Q.build_kernels(qcfg, X, Ha, Hd, chrom, keep_xc=False)
    Xfix = np.ones((n, 1))

    gm = GeneticModel(X, Ha, Hd, chrom, K, meta, y, Xfix)
    say("    variance proportions: " +
        "  ".join(f"{k}={v:.4f}" for k, v in gm.prop.items()))
    if not gm.use_epi:
        say("    [note] the epistatic component was estimated at ~0; "
            "ADD+DOM+EPI will coincide with ADD+DOM.")

    out = []
    for rep in range(cfg.N_REPS):
        rng = np.random.default_rng(cfg.SEED + rep * 31)
        cand = np.sort(rng.choice(n, min(cfg.N_CANDIDATES, n), replace=False))
        G0 = X[cand].copy()
        KA_cand = meta["KA_raw"][np.ix_(cand, cand)]
        say(f"  replicate {rep+1}/{cfg.N_REPS}: {len(cand)} founders, "
            f"{len(cand)*(len(cand)-1)//2:,} crosses per generation")
        t1 = time.time()
        out.append(simulate_multigen(cfg, gm, G0, KA_cand, cfg.TRAIT, rep, rng))
        say(f"    done in {(time.time()-t1)/60:.1f} min")
    D = pd.concat(out, ignore_index=True)
    f = os.path.join(cfg.OUT_DIR, f"{cfg.TRAIT}_multigen.csv")
    D.to_csv(f, index=False)

    g = (D.groupby(["strategy", "generation"])
         .agg(mean_total=("mean_total", "mean"), sd=("mean_total", "std"),
              sd_total=("sd_total", "mean"), mean_F=("mean_F", "mean"))
         .reset_index())
    say("\n" + "=" * 70)
    say("Realised genetic gain by generation (mean over replicates)")
    say("=" * 70)
    say(g.pivot_table(index="generation", columns="strategy",
                      values="mean_total").round(4).to_string())
    say("\nInbreeding (mean diagonal of the realised relationship matrix - 1)")
    say(g.pivot_table(index="generation", columns="strategy",
                      values="mean_F").round(4).to_string())
    g.to_csv(os.path.join(cfg.OUT_DIR, f"{cfg.TRAIT}_multigen_summary.csv"), index=False)
    say(f"\nDone in {(time.time()-t0)/60:.1f} min -> {cfg.OUT_DIR}/")
    return D


# ============================================================================
# [F] figures
# ============================================================================
def fig_accuracy(res_dirs, out):
    rows = []
    for d in res_dirs:
        for f in glob.glob(os.path.join(d, "*_accuracy_bias.csv")):
            rows.append(pd.read_csv(f))
    if not rows:
        return None
    A = pd.concat(rows, ignore_index=True)
    keep = ["A", "A+D", "A+D+MULT_t", "A+D+HAD"]
    A = A[A.model.isin(keep)]
    traits = sorted(A.trait.unique())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    w = 0.8 / len(keep)
    for i, mdl in enumerate(keep):
        sub = A[A.model == mdl].set_index("trait").reindex(traits)
        x = np.arange(len(traits)) + i * w
        ax[0].bar(x, sub.accuracy, w, yerr=sub.acc_sd, capsize=2, label=mdl)
        ax[1].bar(x, sub.bias, w, yerr=sub.bias_sd, capsize=2, label=mdl)
    for a, t, yl in [(ax[0], "Predictive ability", "cor(GEBV, adjusted phenotype)"),
                     (ax[1], "Prediction bias", "regression slope")]:
        a.set_xticks(np.arange(len(traits)) + 0.4 - w / 2)
        a.set_xticklabels(traits)
        a.set_title(t)
        a.set_ylabel(yl)
    ax[1].axhline(1.0, ls="--", c="k", lw=0.8)
    ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = os.path.join(out, "Fig1_accuracy_bias.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_delta(res_dirs, out):
    rows = []
    for d in res_dirs:
        f = os.path.join(d, "diffuse_delta_pcc.csv")
        if os.path.exists(f):
            rows.append(pd.read_csv(f))
    if not rows:
        return None
    D = pd.concat(rows, ignore_index=True)
    g = (D.groupby(["trait", "model"])
         .agg(d=("dPCC", "mean"),
              pos=("dPCC", lambda v: float(np.mean(np.asarray(v) > 0))))
         .reset_index())
    piv = g.pivot(index="model", columns="trait", values="d")
    pos = g.pivot(index="model", columns="trait", values="pos")
    fig, ax = plt.subplots(figsize=(1.1 * piv.shape[1] + 3, 0.35 * piv.shape[0] + 2))
    v = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if np.isfinite(piv.values[i, j]):
                star = "*" if pos.values[i, j] >= 0.95 else ""
                ax.text(j, i, f"{piv.values[i,j]:+.3f}{star}", ha="center",
                        va="center", fontsize=6)
    ax.set_title("Delta predictive ability vs A+D\n(* = positive in >=95% of splits)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    p = os.path.join(out, "Fig2_delta_pcc_heatmap.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_multigen(files, out):
    rows = [pd.read_csv(f) for f in files if os.path.exists(f)]
    if not rows:
        return None
    D = pd.concat(rows, ignore_index=True)
    traits = sorted(D.trait.unique())
    fig, axes = plt.subplots(2, len(traits), figsize=(3.1 * len(traits), 5.4),
                             squeeze=False)
    for k, tr in enumerate(traits):
        sub = D[D.trait == tr]
        for strat in STRATEGIES:
            s = sub[sub.strategy == strat].groupby("generation")
            m, e = s.mean_total.mean(), s.mean_total.std()
            axes[0][k].plot(m.index, m.values, "-o", ms=3, color=PALETTE[strat],
                            label=strat)
            axes[0][k].fill_between(m.index, m - e, m + e, color=PALETTE[strat],
                                    alpha=0.15, lw=0)
            f = sub[sub.strategy == strat].groupby("generation").mean_F.mean()
            axes[1][k].plot(f.index, f.values, "-o", ms=3, color=PALETTE[strat])
        axes[0][k].set_title(tr)
        axes[0][k].set_ylabel("mean genetic value" if k == 0 else "")
        axes[1][k].set_ylabel("inbreeding" if k == 0 else "")
        axes[1][k].set_xlabel("generation")
    axes[0][0].legend(fontsize=6)
    fig.suptitle("Realised gain and inbreeding over five generations of mate allocation",
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(out, "Fig3_multigen_gain.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_kernels(res_dirs, out):
    f = None
    for d in res_dirs:
        c = os.path.join(d, "kernel_collinearity.csv")
        if os.path.exists(c):
            f = c
            break
    if f is None:
        return None
    CM = pd.read_csv(f, index_col=0)
    fig, ax = plt.subplots(figsize=(0.45 * len(CM) + 2.5, 0.45 * len(CM) + 2))
    im = ax.imshow(CM.values, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(CM)))
    ax.set_xticklabels(CM.columns, rotation=90)
    ax.set_yticks(range(len(CM)))
    ax.set_yticklabels(CM.index)
    for i in range(len(CM)):
        for j in range(len(CM)):
            ax.text(j, i, f"{CM.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=5, color="w")
    ax.set_title("Kernel correlation\n(|r|>0.5: variance components not separable)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    p = os.path.join(out, "FigS1_kernel_collinearity.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_mating(res_dirs, out):
    rows = []
    for d in res_dirs:
        f = os.path.join(d, "mating_summary.csv")
        if os.path.exists(f):
            rows.append(pd.read_csv(f))
    if not rows:
        return None
    M = pd.concat(rows, ignore_index=True)
    traits = sorted(M.trait.unique())
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for i, strat in enumerate(sorted(M.strategy.unique())):
        sub = M[M.strategy == strat].set_index("trait").reindex(traits)
        x = np.arange(len(traits)) + i * 0.25
        ax[0].bar(x, sub.mean_gv_full, 0.25, label=strat)
        ax[1].bar(x, sub.overlap_with_ADD, 0.25, label=strat)
    for a, t in [(ax[0], "Expected progeny value of the selected plan"),
                 (ax[1], "Overlap with the additive-only plan")]:
        a.set_xticks(np.arange(len(traits)) + 0.25)
        a.set_xticklabels(traits)
        a.set_title(t)
    ax[1].set_ylim(0, 1.05)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(out, "Fig4_mating_plans.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def make_figures(res_glob, multigen_glob, out):
    os.makedirs(out, exist_ok=True)
    dirs = sorted(glob.glob(res_glob))
    say(f"[figures] scanning {len(dirs)} result directories")
    made = []
    for fn, args in [(fig_accuracy, (dirs, out)), (fig_delta, (dirs, out)),
                     (fig_kernels, (dirs, out)), (fig_mating, (dirs, out)),
                     (fig_multigen, (sorted(glob.glob(multigen_glob)), out))]:
        try:
            p = fn(*args)
            if p:
                made.append(p)
                say(f"    wrote {os.path.basename(p)}")
            else:
                say(f"    skipped {fn.__name__} (inputs not found)")
        except Exception as e:
            say(f"    {fn.__name__} failed: {type(e).__name__}: {e}")
    return made


# ============================================================================
# Self-test
# ============================================================================
def selftest():
    say("=" * 74)
    say("SELF-TEST")
    say("=" * 74)
    ok = True
    X = P.synth_data(seed=3, n=400, per_chr=60)[0]
    n, m = X.shape
    chrom = np.repeat(np.arange(1, m // 60 + 2), 60)[:m]
    Ha, Hd = P.noia(X)
    qcfg = Q.CFG()
    qcfg.CACHE_DIR = "/tmp/mgck"
    qcfg.DO_CIS = False
    K, meta = Q.build_kernels(qcfg, X, Ha, Hd, chrom, keep_xc=False)
    rng = np.random.default_rng(0)
    d, U = np.linalg.eigh(K["A"])
    g = U @ (np.sqrt(np.maximum(d, 0)) * rng.normal(0, 1, n))
    de, Ue = np.linalg.eigh(K["MULT_t"])
    e = Ue @ (np.sqrt(np.maximum(de, 0)) * rng.normal(0, 1, n))
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    y = 0.6 * z(g) + 0.4 * z(e) + 0.6 * rng.normal(0, 1, n)
    gm = GeneticModel(X, Ha, Hd, chrom, K, meta, y, np.ones((n, 1)))
    say(f"\n[1] variance proportions: " +
        "  ".join(f"{k}={v:.3f}" for k, v in gm.prop.items()))

    say("\n[2] expected vs simulated offspring value (should agree)")
    r = np.random.default_rng(1)
    i, j = 3, 17
    e_ha, e_hd = gm.expected_codes(X[i:i + 1], X[j:j + 1])
    exp_add = float((e_ha @ gm.a_hat)[0])
    exp_dom = float((e_hd @ gm.d_hat)[0])
    exp_epi = float(gm.epi_value(e_ha)[0])
    ga = sample_gametes(X[i], 4000, r)
    gb = sample_gametes(X[j], 4000, r)
    kid = ga + gb
    v = gm.value(kid)
    # Compare against the Monte-Carlo standard error, not a relative deviation:
    # these values are close to zero, so a relative criterion is dominated by
    # simulation noise rather than by any real discrepancy.
    for nm, exp, arr in [("additive", exp_add, v["add"]),
                         ("dominance", exp_dom, v["dom"]),
                         ("epistatic", exp_epi, v["epi"])]:
        sim = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(len(arr))) + 1e-12
        zsc = abs(exp - sim) / se
        say(f"    {nm:10s} expected {exp:+.5f}  simulated {sim:+.5f}  "
            f"MC s.e. {se:.5f}  z = {zsc:.2f}")
        if zsc > 4.0:
            ok = False
            say(f"      [FAIL] {nm} deviates by more than 4 standard errors")

    say("\n[3] short simulation runs and gain is monotone for ADD")
    cfg = MCFG()
    cfg.N_CANDIDATES, cfg.N_CROSSES = 40, 8
    cfg.OFF_PER_CROSS, cfg.N_SCREEN, cfg.GENERATIONS = 4, 12, 2
    cand = np.arange(40)
    D = simulate_multigen(cfg, gm, X[cand].copy(),
                          meta["KA_raw"][np.ix_(cand, cand)], "test", 0,
                          np.random.default_rng(0))
    piv = D.pivot_table(index="generation", columns="strategy", values="mean_total")
    say(piv.round(4).to_string())
    add_traj = piv["ADD"].values
    if not np.all(np.diff(add_traj) > -1e-9):
        ok = False
        say("    [FAIL] ADD gain is not monotone")
    if piv["ADD"].iloc[-1] <= piv["RANDOM"].iloc[-1]:
        ok = False
        say("    [FAIL] ADD does not beat RANDOM")

    say("\n[4] figure generation")
    os.makedirs("/tmp/mgfig", exist_ok=True)
    D.to_csv("/tmp/mgfig/test_multigen.csv", index=False)
    made = make_figures("/tmp/nonexistent_*", "/tmp/mgfig/*_multigen.csv", "/tmp/mgfig")
    if not any("Fig3" in p for p in made):
        ok = False
        say("    [FAIL] multigen figure not produced")

    say("\nSELF-TEST: " + ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile")
    ap.add_argument("--trait")
    ap.add_argument("--out")
    ap.add_argument("--stage", choices=["multigen", "figures"], default="multigen")
    ap.add_argument("--generations", type=int)
    ap.add_argument("--reps", type=int)
    ap.add_argument("--candidates", type=int)
    ap.add_argument("--results_glob", default="pig_results_*")
    ap.add_argument("--multigen_glob", default="pig_multigen_out/*_multigen.csv")
    ap.add_argument("--fig_dir", default="figures")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    cfg = MCFG()
    for v, k in [(a.bfile, "BFILE"), (a.trait, "TRAIT"), (a.out, "OUT_DIR"),
                 (a.generations, "GENERATIONS"), (a.reps, "N_REPS"),
                 (a.candidates, "N_CANDIDATES")]:
        if v is not None:
            setattr(cfg, k, v)
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.stage == "multigen":
        run_multigen(cfg)
    else:
        make_figures(a.results_glob, a.multigen_glob, a.fig_dir)


if __name__ == "__main__":
    main()
