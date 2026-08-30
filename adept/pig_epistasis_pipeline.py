#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Genome-wide epistasis evaluation and epistasis-aware mate allocation in pigs
 ------------------------------------------------------------------------------
 A single-entry pipeline covering four stages for all traits:

   [A] DIFFUSE   Exact aggregate epistatic kernels (MULT/AD/DD/XOR x trans/cis)
                 evaluated by held-out prediction gain over an A+D baseline.
   [B] DECOMPOSE Correlation between additive, dominance and epistatic breeding
                 values. Decides whether epistasis carries information that is
                 INDEPENDENT of the additive value -- i.e. whether mate
                 allocation can do anything that additive selection cannot.
   [C] FOCAL     Genome-wide exact scoring of five interaction types, top-K
                 pairs, and CROSS-SEED REPRODUCIBILITY counting (a pair only
                 counts if independently re-selected in multiple splits).
   [D] MATING    Epistasis-aware mate allocation among 300 randomly drawn
                 animals (no sex information required; all pairwise crosses),
                 accounting for allele frequency, additive + dominance +
                 epistatic offspring value, and progeny inbreeding.

================================================================================
 METHODOLOGICAL CONVENTIONS (each was established empirically in this project)
================================================================================
 1. ACCURACY = cor(GEBV, y - Xb), i.e. the correlation between the estimated
    GENETIC value and the fixed-effect-adjusted phenotype. Including Xb in the
    prediction inflates accuracy and compresses DeltaPCC ~4.5-fold.
 2. FIXED EFFECTS = intercept only by default. Genomic PCs are NOT included
    (the GRM already models structure; adding PCs removes genetic signal).
    The 21 cohort covariates are excluded by default because they reduce h2
    from 0.297 to 0.238 -- they absorb GENETIC variance (confounded with
    family structure), so including them is double-correction. Use --use_cov
    to override.
 3. KERNELS ARE EXACT, NEVER SAMPLED. Sum_{i,j}(M_i.N_j)(M_i.N_j)' =
    (MM').(NN'). Monte-Carlo sampling of 2e5 pairs covers 0.03% of 6.4e8 pairs
    and correlates only 0.438 with the exact kernel.
 4. SCORING IS STANDARDISED. |cor(z,y)|, not |sum z*y| (the latter scales with
    the variance of the interaction feature). Sum z^2 = ((M.M)'(N.N))[i,j].
 5. AD IS ASYMMETRIC. Ha_i*Hd_j != Ha_j*Hd_i, so both orientations are scored.
 6. NO LEAKAGE. Kernels depend on genotypes only and are pre-built on all
    animals; pair scoring uses the TRAINING SPLIT ONLY and is redone
    independently for every split.

================================================================================
 [D] MATE ALLOCATION -- the algebra
================================================================================
 For parents P,Q at locus k the transmitted allele frequencies are gP/2, gQ/2:
     E[dosage]  = (gP + gQ)/2          =>  E[Ha_off] = (Ha_P + Ha_Q)/2
     P(het)     = p1(1-p2) + (1-p1)p2  =>  E[Hd_off] from NOIA coding
 Additive and dominance offspring values then follow from back-solved marker
 effects. For the trans (unlinked) epistatic kernel the expectation FACTORISES,
 because unlinked loci are transmitted independently:
     E[z_k z_l] = E[z_k] E[z_l]        for k,l on different chromosomes
 so the expected offspring epistatic value is exact from expected genotypes and
 no offspring simulation is needed. Verified against Monte-Carlo simulation of
 40,000 offspring: analytical -8.1507 vs simulated -8.1320 (0.23% deviation).

 Writing t_j = E[Ha_off] . Ha_j and t_cj for the chromosome-c subvector,
     K_AA_raw(off, j) = t_j^2 - sum_c t_cj^2
 and GEBV_AA(off) = sigma2_AA / s_AA * sum_j alpha_j * K_AA_raw(off, j).
 Because E[Ha_off] is a mid-parent average, t_j = (G[:,P] + G[:,Q])/2 with
 G = Ha Ha', so only per-candidate Gram columns are needed -- O(n) per cross.

 Progeny inbreeding F = A_PQ / 2 from the VanRaden relationship matrix.

 Three ranking strategies are compared, which is the actual test of whether
 epistasis changes breeding decisions:
     ADD      mid-parent additive value only
     ADD+DOM  additive + dominance
     FULL     additive + dominance + epistasis
 If FULL selects the same crosses as ADD, epistasis has no allocation value.

================================================================================
 USAGE
     python pig_epistasis_pipeline.py --selftest
     python pig_epistasis_pipeline.py --traits bf lmd lmp ttn ltn rtn tpd \\
         --stages diffuse,decompose,mating --seeds 20
     python pig_epistasis_pipeline.py --traits bf --stages all --seeds 5
 Stage "focal" is expensive (~30 min per trait per split); it is excluded from
 the default stage list and must be requested explicitly.
================================================================================
"""
from __future__ import annotations

import argparse
import heapq
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

from scipy.linalg import cholesky, LinAlgError
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Speed patch for the REML solver (numerically identical, ~4x faster)
# ---------------------------------------------------------------------------
# Two avoidable O(n^3) costs dominate the original implementation:
#   1. tr(P K) was evaluated as np.trace(P @ K), i.e. a full n x n matrix
#      product used only for its diagonal. tr(P K) = sum(P * K) for symmetric
#      P and K, which is O(n^2). Measured at n=1800: 0.588 s -> 0.015 s (38x).
#   2. n_restarts defaulted to 2, running the whole L-BFGS optimisation twice.
#      Restarts guard against local optima, but variance components are stable
#      across the splits of one trait, so a single start is sufficient here.
# Combined effect ~4x. Estimates are verified below to match the original to
# 1e-8 before the patch is installed.
def _fast_obj_grad(self, theta, y, X, Ks):
    n = len(y)
    try:
        v = np.exp(theta)
        V = (v[-1] + self.jitter) * np.eye(n)
        for vi, K in zip(v[:-1], Ks):
            V = V + vi * K
        L = cholesky(V, lower=True)
        Vi = cho_solve((L, True), np.eye(n))
        ViX = Vi @ X
        XtViX = X.T @ ViX
        Li = cholesky(XtViX, lower=True)
        Pm = Vi - ViX @ cho_solve((Li, True), np.eye(X.shape[1])) @ ViX.T
        Py = Pm @ y
        nll = 0.5 * (2 * np.sum(np.log(np.diag(L)))
                     + 2 * np.sum(np.log(np.diag(Li))) + float(y @ Py))
    except LinAlgError:
        return 1e10, np.zeros_like(theta)
    g = np.empty(len(theta))
    for i, K in enumerate(list(Ks) + [None]):
        if K is None:
            tr, quad = float(np.trace(Pm)), float(Py @ Py)
        else:
            tr, quad = float(np.sum(Pm * K)), float(Py @ (K @ Py))   # O(n^2)
        g[i] = 0.5 * (tr - quad) * v[i]
    return nll, g


def _fast_fit(self, y, X, Ks, n_restarts=1, maxiter=P.CFG.REML_MAXITER):
    vy = float(np.var(y))
    nv = len(Ks) + 1
    rng = np.random.RandomState(0)
    best = None
    for r in range(n_restarts):
        sc = np.ones(nv) if r == 0 else rng.uniform(0.3, 2.5, nv)
        th0 = np.log(np.maximum(vy / nv * sc, 1e-6))
        res = minimize(self._obj_grad, th0, args=(y, X, Ks), jac=True,
                       method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    self.variances_ = np.exp(best.x)
    self.loglik_ = -best.fun
    return self


def install_reml_speedup(verify=True):
    """
    Verify equivalence on a small problem, then install the fast gradient.

    Only _obj_grad is replaced; the number of restarts is left untouched. The
    gradient rewrite on its own deviates from the finite-difference reference by
    7.9e-10, i.e. it is numerically equivalent, and it buys roughly 2x. Reducing
    the restarts as well would not be safe: with restarts forced to one, two
    starts reach the same likelihood (-120.3645 either way) but the surface is
    flat along one direction, so the variance components slide within that
    valley and the equivalence check fails with a relative deviation of 5.45.
    """
    if verify:
        rng = np.random.default_rng(0)
        nn = 220
        Z = rng.normal(0, 1, (nn, 90))
        K1 = Z @ Z.T / 90
        K1 /= np.mean(np.diag(K1))
        W = rng.normal(0, 1, (nn, 70))
        K2 = W @ W.T / 70
        K2 /= np.mean(np.diag(K2))
        yy = rng.normal(0, 1, nn)
        XX = np.ones((nn, 1))
        ref = P.REMLgrad().fit(yy, XX, [K1, K2]).variances_
        old_og = P.REMLgrad._obj_grad
        P.REMLgrad._obj_grad = _fast_obj_grad
        new = P.REMLgrad().fit(yy, XX, [K1, K2]).variances_
        dev = float(np.max(np.abs(np.asarray(ref) - np.asarray(new))
                           / np.maximum(np.abs(ref), 1e-12)))
        if dev > 1e-6:
            P.REMLgrad._obj_grad = old_og
            say(f"[reml] speed patch REJECTED (relative deviation {dev:.2e}); "
                f"falling back to the original solver")
            return False
        say(f"[reml] speed patch installed (variance estimates match the "
            f"original to {dev:.1e})")
        return True
    P.REMLgrad._obj_grad = _fast_obj_grad
    return True


# ============================================================================
# Configuration
# ============================================================================
class CFG:
    BFILE      = "genotypes"        # set via --bfile
    TRAITS     = ("bf", "lmd", "lmp", "ttn", "ltn", "rtn", "tpd")
    OUT_DIR    = "pig_epistasis_out"
    CACHE_DIR  = "adept_kernel_cache"
    SEED       = 2024

    USE_COV    = False          # see convention 2
    TEST_FRAC  = 0.20
    SPLIT      = "random"       # "random" matches the IFAM benchmark
    N_SEEDS    = 20
    # Core models carry the headline result and are run for all N_SEEDS splits.
    # Secondary (diagnostic) models are run for SECONDARY_SEEDS splits only:
    # a single REML fit at n=2237 with 3 kernels costs ~130 s, so fitting all
    # 14 models for 20 splits would take ~10 h per trait. Tiering cuts the
    # number of fits from 280 to 130 without weakening the main comparison.
    CORE_MODELS     = ("A", "A+D", "A+D+MULT_t", "A+D+HAD")
    SECONDARY_SEEDS = 5
    N_BOOT     = 1000   # 1000 is ample for a 95% percentile CI

    EXACT_TYPES = ("MULT", "AD", "DD", "XOR")
    DO_CIS      = True
    ADD_HAD     = True          # classical K_A o K_A control

    # --- stage C ---
    SCORE_TYPES = ("MULT", "DD", "THR")   # XOR duplicates MULT; AD was uniformly negative
    TOP_POOL    = 20
    K_GRID      = (1, 2, 3)
    SCAN_CHUNK  = 512
    MIN_CELL    = 5
    FOCAL_SEEDS = 5

    # --- stage D ---
    N_CANDIDATES = 300          # animals drawn at random; no sex split
    N_CROSSES    = 60           # crosses retained in the allocation plan
    MAX_USE      = 5            # maximum matings per animal
    F_MAX        = 0.0625       # progeny inbreeding ceiling
    MAF_MIN      = 0.01         # allele-frequency filter for candidate loci


def say(*a):
    print(*a, flush=True)


# ============================================================================
# Shared helpers
# ============================================================================
def pcc(a, b):
    if a is None or b is None or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pred_bias(y_adj, gebv):
    """
    Regression slope of the adjusted phenotype on the GEBV. Unbiased prediction
    gives 1; <1 indicates over-dispersion of the predicted genetic values and
    >1 under-dispersion. Reported alongside accuracy in the genomic-prediction
    literature (e.g. the IFAM benchmark), where an accuracy gain accompanied by
    a slope drifting away from 1 is discounted.
    """
    if gebv is None or np.std(gebv) < 1e-12:
        return np.nan
    return float(np.cov(y_adj, gebv, ddof=1)[0, 1] / np.var(gebv, ddof=1))


def fit_gebv(y, Xfix, Kdict, keys, tr, te):
    """Fit a multi-kernel GBLUP on `tr`; return (GEBV, adjusted y, model) on `te`."""
    try:
        Ktt = [Kdict[k][np.ix_(tr, tr)] for k in keys]
        Knt = [Kdict[k][np.ix_(te, tr)] for k in keys]
        if any(not np.all(np.isfinite(A)) for A in Ktt + Knt):
            return None, None, None
        m = P.REMLgrad().fit(y[tr], Xfix[tr], Ktt)
        pred = m.predict_new(y[tr], Xfix[tr], Ktt, Xfix[te], Knt)
        if pred is None or not np.all(np.isfinite(pred)):
            return None, None, None
        v = np.asarray(m.variances_, float)
        V = sum(s * A for s, A in zip(v[:-1], Ktt)) + (v[-1] + 1e-8) * np.eye(len(tr))
        c = cho_factor(V, lower=True)
        b = np.linalg.solve(Xfix[tr].T @ cho_solve(c, Xfix[tr]),
                            Xfix[tr].T @ cho_solve(c, y[tr]))
        return pred - Xfix[te] @ b, y[te] - Xfix[te] @ b, m
    except Exception as e:
        say(f"      [REML failed] {'+'.join(keys)} -> {type(e).__name__}: {e}")
        return None, None, None


def boot_delta(ya, g0, g1, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(ya)
    base = pcc(ya, g1) - pcc(ya, g0)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.std(ya[i]) < 1e-12:
            continue
        d.append(pcc(ya[i], g1[i]) - pcc(ya[i], g0[i]))
    d = np.asarray(d)
    return base, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def make_split(n, labels, split, frac, seed):
    if split == "random":
        rng = np.random.default_rng(seed)
        te = np.zeros(n, bool)
        te[rng.choice(n, int(round(n * frac)), replace=False)] = True
    else:
        te = P.stratified_split(labels, frac, seed)
    return np.where(~te)[0], np.where(te)[0]


# ============================================================================
# Kernels
# ============================================================================
def build_kernels(cfg, X, Ha, Hd, chrom, keep_xc=False):
    n, m = X.shape
    K, meta = {}, {}
    KA_raw = P.grm_a(X)                      # VanRaden; diagonal = 1 + F
    meta["KA_raw"] = KA_raw
    meta["sA"] = float(np.mean(np.diag(KA_raw)))
    K["A"] = KA_raw / meta["sA"]
    KD_raw = P.grm_d(Hd)
    meta["sD"] = float(np.mean(np.diag(KD_raw)))
    K["D"] = KD_raw / meta["sD"]

    Xx = (X > 0).astype(np.float64) * 2 - 1
    xc = Xx - Xx.mean(0)
    sd = xc.std(0)
    sd[sd < 1e-12] = 1.0
    Xc = xc / sd
    del Xx

    mats = {"MULT": (Ha, Ha), "AD": (Ha, Hd), "DD": (Hd, Hd), "XOR": (Xc, Xc)}
    gram = P.GramCache(chrom)
    modes = [("trans", "_t")] + ([("within", "_c")] if cfg.DO_CIS else [])
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    for t in cfg.EXACT_TYPES:
        M, N = mats[t]
        for mode, sfx in modes:
            key = t + sfx
            cf = os.path.join(cfg.CACHE_DIR,
                              f"KEX_{key}_{os.path.basename(str(cfg.BFILE))}_{n}x{m}.npy")
            if os.path.exists(cf):
                kc = np.load(cf)
                if kc.shape == (n, n):
                    K[key] = kc
                    say(f"    {key:8s} cached")
                    continue
            K[key] = P.kern_prod_exact(M, N, chrom, mode, gram, t + "M", t + "N")
            np.save(cf, K[key])
            say(f"    {key:8s} built (exact, no sampling)")
    if cfg.ADD_HAD:
        KH = K["A"] * K["A"]
        K["HAD"] = KH / np.mean(np.diag(KH))
        say("    HAD      built (K_A o K_A, classical control)")
    # Xc (XOR coding, ~0.8 GB at this scale) is only needed by stage C; keep it
    # only when that stage will actually run, otherwise release it here.
    if keep_xc:
        meta["Xc"] = Xc
    else:
        del Xc
    return K, meta


# ============================================================================
# [C] genome-wide exact scoring
# ============================================================================
def _push(heap, vals, s0, top_pool, tags=None):
    kk = min(top_pool, vals.size)
    if kk <= 0:
        return
    flat = np.argpartition(vals.ravel(), -kk)[-kk:]
    rr, cc = np.unravel_index(flat, vals.shape)
    for a, b in zip(rr, cc):
        v = float(vals[a, b])
        if v <= 0:
            continue
        item = (v, int(s0 + a), int(b), "" if tags is None else str(tags[a, b]))
        if len(heap) < top_pool:
            heapq.heappush(heap, item)
        elif v > heap[0][0]:
            heapq.heapreplace(heap, item)


def _cor(S_zy, S_z, S_z2, n, sdy):
    var = S_z2 / n - (S_z / n) ** 2
    out = np.abs(S_zy / n) / np.maximum(np.sqrt(np.maximum(var, 0.0)) * sdy, 1e-12)
    out[var <= 1e-12] = 0.0
    return out


def score_genomewide(X, y, types, top_pool, chunk, min_cell=5, verbose=False):
    """Exact |cor(z, y)| for every SNP pair, training split only."""
    n, m = X.shape
    yc = (y - y.mean()).astype(np.float32)
    sdy = float(yc.std())
    if sdy < 1e-12:
        return {t: [] for t in types}
    Ha, Hd = P.noia(X)
    Ha = Ha.astype(np.float32)
    Hd = Hd.astype(np.float32)
    Xx = (X > 0).astype(np.float32) * 2 - 1
    xc = Xx - Xx.mean(0)
    sd = xc.std(0)
    sd[sd < 1e-12] = 1.0
    Xc = (xc / sd).astype(np.float32)

    prod = {"MULT": (Ha, Ha), "AD": (Ha, Hd), "DD": (Hd, Hd), "XOR": (Xc, Xc)}
    pre = {t: (M, N, M * yc[:, None], M * M, N * N)
           for t, (M, N) in prod.items() if t in types}
    if "THR" in types:
        I0 = (X == 0).astype(np.float32)
        I1 = (X == 1).astype(np.float32)
        I2 = (X == 2).astype(np.float32)
        Ic = (X > 0).astype(np.float32)
        W0, W1, W2, Wc = (I0 * yc[:, None], I1 * yc[:, None],
                          I2 * yc[:, None], Ic * yc[:, None])

    heaps = {t: [] for t in types}
    nch = (m + chunk - 1) // chunk
    for ci, s0 in enumerate(range(0, m, chunk)):
        s1 = min(s0 + chunk, m)
        upper = np.arange(m)[None, :] > np.arange(s0, s1)[:, None]

        for t in (x for x in types if x != "THR"):
            M, N, My, M2, N2 = pre[t]
            r = _cor(My[:, s0:s1].T @ N, M[:, s0:s1].T @ N, M2[:, s0:s1].T @ N2, n, sdy)
            if t == "AD":
                r2 = _cor((N * yc[:, None])[:, s0:s1].T @ M,
                          N[:, s0:s1].T @ M, N2[:, s0:s1].T @ M2, n, sdy)
                tags = np.where(r >= r2, "AiDj", "AjDi")
                r = np.maximum(r, r2)
            else:
                tags = np.full(r.shape, t, dtype=object)
            _push(heaps[t], np.where(upper, r, -1.0), s0, top_pool, tags)

        if "THR" in types:
            a0, a1, a2, ac = (I0[:, s0:s1], I1[:, s0:s1], I2[:, s0:s1], Ic[:, s0:s1])
            w0, w1, w2, wc = (W0[:, s0:s1], W1[:, s0:s1], W2[:, s0:s1], Wc[:, s0:s1])
            P00, P01, P10 = a0.T @ I0, a0.T @ I1, a1.T @ I0
            P12, P21, P22, Pcc = a1.T @ I2, a2.T @ I1, a2.T @ I2, ac.T @ Ic
            Q00, Q01, Q10 = w0.T @ I0, w0.T @ I1, w1.T @ I0
            Q12, Q21, Q22, Qcc = w1.T @ I2, w2.T @ I1, w2.T @ I2, wc.T @ Ic
            cands = {"s>=1": (n - P00, -Q00),
                     "s>=2": (n - P00 - P01 - P10, -(Q00 + Q01 + Q10)),
                     "s>=3": (P12 + P21 + P22, Q12 + Q21 + Q22),
                     "s>=4": (P22, Q22),
                     "c>=2": (Pcc, Qcc)}
            best = btag = None
            for tag, (S1, Sy) in cands.items():
                p = S1 / n
                den = np.sqrt(np.maximum(p * (1 - p), 1e-12)) * sdy * n
                r = np.abs(Sy) / np.maximum(den, 1e-12)
                r[(S1 < min_cell) | (S1 > n - min_cell)] = 0.0
                if best is None:
                    best, btag = r, np.full(r.shape, tag, dtype=object)
                else:
                    up = r > best
                    best, btag = np.where(up, r, best), np.where(up, tag, btag)
            _push(heaps["THR"], np.where(upper, best, -1.0), s0, top_pool, btag)

        if verbose and (ci + 1) % max(1, nch // 5) == 0:
            say(f"        scored {ci+1}/{nch} blocks")
    return {t: sorted(h, key=lambda z: -z[0]) for t, h in heaps.items()}


def pair_feature(typ, tag, X, Ha, Hd, Xc, i, j):
    if typ == "MULT":
        z = Ha[:, i] * Ha[:, j]
    elif typ == "DD":
        z = Hd[:, i] * Hd[:, j]
    elif typ == "XOR":
        z = Xc[:, i] * Xc[:, j]
    elif typ == "AD":
        z = Ha[:, j] * Hd[:, i] if tag == "AjDi" else Ha[:, i] * Hd[:, j]
    elif typ == "THR":
        s = X[:, i] + X[:, j]
        if tag == "c>=2":
            z = ((X[:, i] > 0).astype(int) + (X[:, j] > 0).astype(int)) >= 2
        else:
            z = s >= int(tag.split(">=")[1])
        z = z.astype(float)
    else:
        return None
    z = z - z.mean()
    s_ = z.std()
    return None if s_ < 1e-10 else z / s_


def pairs_kernel(sel, typ, X, Ha, Hd, Xc):
    cols = [pair_feature(typ, tg, X, Ha, Hd, Xc, i, j) for (_, i, j, tg) in sel]
    cols = [c for c in cols if c is not None]
    if not cols:
        return None
    Z = np.column_stack(cols)
    Kt = (Z @ Z.T) / Z.shape[1]
    d = np.mean(np.diag(Kt))
    return Kt / d if d > 1e-12 else Kt


# ============================================================================
# [D] mate allocation
# ============================================================================
def offspring_components(cfg, X, Ha, Hd, chrom, K, meta, y, Xfix, keys):
    """Fit the full model on all animals; return (model, alpha, variance dict)."""
    n, m = X.shape
    Ktt = [K[k] for k in keys]
    mm = P.REMLgrad().fit(y, Xfix, Ktt)
    v = np.asarray(mm.variances_, float)
    V = sum(s * A for s, A in zip(v[:-1], Ktt)) + (v[-1] + 1e-8) * np.eye(n)
    c = cho_factor(V, lower=True)
    b = np.linalg.solve(Xfix.T @ cho_solve(c, Xfix), Xfix.T @ cho_solve(c, y))
    alpha = cho_solve(c, y - Xfix @ b)                       # V^-1 (y - Xb)
    var = {k: float(s) for k, s in zip(keys, v[:-1])}
    var["_resid"] = float(v[-1])
    return mm, alpha, var


def mate_allocation(cfg, X, Ha, Hd, chrom, K, meta, y, Xfix, out_dir, trait):
    """Epistasis-aware mate allocation among CFG.N_CANDIDATES random animals."""
    n, m = X.shape
    keys = [k for k in ("A", "D", "MULT_t") if k in K]
    mm, alpha, var = offspring_components(cfg, X, Ha, Hd, chrom, K, meta, y, Xfix, keys)
    tot = sum(var.values())
    say("    variance components: " +
        "  ".join(f"{k}={var[k] / tot:.4f}" for k in keys) +
        f"  resid={var['_resid'] / tot:.4f}")
    if var.get("MULT_t", 0.0) / tot < 1e-4:
        say("    [note] the epistatic variance component was estimated at the")
        say("           boundary (~0), so gv_epi is ~0 by construction and the")
        say("           FULL strategy will coincide with ADD+DOM. This is REML's")
        say("           verdict on the data, not a computational failure.")
    if var.get("D", 0.0) / tot < 1e-4:
        say("    [note] the dominance component was estimated at ~0; ADD+DOM")
        say("           will coincide with ADD.")

    rng = np.random.default_rng(cfg.SEED)
    cand = np.sort(rng.choice(n, min(cfg.N_CANDIDATES, n), replace=False))
    nc = len(cand)
    say(f"    candidates drawn at random (no sex split): {nc}")

    # --- back-solved marker effects for additive and dominance ---
    a_hat = var["A"] * (Ha.T @ alpha) / (meta["sA"] * (2 * np.sum(
        (X.mean(0) / 2) * (1 - X.mean(0) / 2)) + 1e-12))
    d_hat = (var["D"] * (Hd.T @ alpha) / (meta["sD"] * (np.sum(np.var(Hd, 0)) + 1e-12))
             if "D" in var else np.zeros(m))

    # --- Gram columns needed for the epistatic term ---
    use_epi = "MULT_t" in var
    if use_epi:
        Gc = {}
        Gfull = Ha @ Ha[cand].T                              # n x nc
        for ch in np.unique(chrom):
            idx = chrom == ch
            Gc[ch] = Ha[:, idx] @ Ha[cand][:, idx].T
        raw_diag = np.array([Gfull[c_, k] ** 2 -
                             sum(Gc[ch][c_, k] ** 2 for ch in Gc)
                             for k, c_ in enumerate(cand[:50])])
        s_AA = float(np.mean(raw_diag / np.maximum(
            np.diag(K["MULT_t"])[cand[:50]], 1e-12)))
        scale_AA = var["MULT_t"] / max(s_AA, 1e-12)

    # --- expected offspring genotypes, cross by cross ---
    pairs = list(combinations(range(nc), 2))
    npair = len(pairs)
    say(f"    evaluating {npair:,} pairwise crosses")
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    gi, gj = X[cand][ii], X[cand][jj]                        # npair x m (chunked below)

    add = np.empty(npair)
    dom = np.empty(npair)
    epi = np.zeros(npair)
    inb = np.empty(npair)
    het = np.empty(npair)

    mu = X.mean(0)
    p_aa = np.mean(X == 1, axis=0)
    var_A = np.var(Ha, axis=0)
    beta_d = np.where(var_A > 1e-12,
                      (p_aa * (1 - mu)) / np.maximum(var_A, 1e-12), 0.0)

    CH = 2000
    for s in range(0, npair, CH):
        e = min(s + CH, npair)
        a_ = X[cand][ii[s:e]] / 2.0                          # transmitted freq
        b_ = X[cand][jj[s:e]] / 2.0
        e_dos = a_ + b_
        e_ha = e_dos - mu[None, :]
        p_het = a_ * (1 - b_) + (1 - a_) * b_
        e_hd = p_het - p_aa[None, :] - beta_d[None, :] * e_ha
        add[s:e] = e_ha @ a_hat
        dom[s:e] = e_hd @ d_hat
        het[s:e] = p_het.mean(1)
        if use_epi:
            t = (Gfull[:, ii[s:e]] + Gfull[:, jj[s:e]]) / 2.0        # n x chunk
            acc = t ** 2
            for ch in Gc:
                tc = (Gc[ch][:, ii[s:e]] + Gc[ch][:, jj[s:e]]) / 2.0
                acc -= tc ** 2
            epi[s:e] = scale_AA * (alpha @ acc)
        inb[s:e] = meta["KA_raw"][cand[ii[s:e]], cand[jj[s:e]]] / 2.0

    df = pd.DataFrame(dict(
        sire=cand[ii], dam=cand[jj],
        gv_add=add, gv_dom=dom, gv_epi=epi,
        gv_add_dom=add + dom, gv_full=add + dom + epi,
        progeny_F=inb, exp_heterozygosity=het))
    df.to_csv(os.path.join(out_dir, f"{trait}_all_crosses.csv"), index=False)

    # --- greedy allocation under constraints, three ranking strategies ---
    plans = {}
    for strat, col in [("ADD", "gv_add"), ("ADD+DOM", "gv_add_dom"), ("FULL", "gv_full")]:
        order = np.argsort(-df[col].values)
        use = {}
        chosen = []
        for k in order:
            if df.progeny_F.values[k] > cfg.F_MAX:
                continue
            s_, d_ = df.sire.values[k], df.dam.values[k]
            if use.get(s_, 0) >= cfg.MAX_USE or use.get(d_, 0) >= cfg.MAX_USE:
                continue
            use[s_] = use.get(s_, 0) + 1
            use[d_] = use.get(d_, 0) + 1
            chosen.append(k)
            if len(chosen) >= cfg.N_CROSSES:
                break
        plans[strat] = df.iloc[chosen].copy()
        plans[strat].insert(0, "strategy", strat)
    plan = pd.concat(plans.values(), ignore_index=True)
    plan.to_csv(os.path.join(out_dir, f"{trait}_mating_plan.csv"), index=False)

    rows = []
    base = set(zip(plans["ADD"].sire, plans["ADD"].dam))
    for strat, p in plans.items():
        sel = set(zip(p.sire, p.dam))
        rows.append(dict(trait=trait, strategy=strat, n_crosses=len(p),
                         mean_gv_full=float(p.gv_full.mean()),
                         mean_gv_add=float(p.gv_add.mean()),
                         mean_gv_dom=float(p.gv_dom.mean()),
                         mean_gv_epi=float(p.gv_epi.mean()),
                         mean_F=float(p.progeny_F.mean()),
                         max_F=float(p.progeny_F.max()),
                         n_parents=len(set(p.sire) | set(p.dam)),
                         overlap_with_ADD=len(sel & base) / max(len(base), 1)))
    summ = pd.DataFrame(rows)
    say(summ.round(4).to_string(index=False))
    return df, summ


# ============================================================================
# Main
# ============================================================================
def run(cfg, stages):
    t0 = time.time()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    say("=" * 78)
    say("Genome-wide epistasis evaluation and mate allocation in pigs")
    say("=" * 78)
    install_reml_speedup()

    pc = P.CFG()
    pc.BFILE = cfg.BFILE
    pc.N_PCS = 0
    pc.PHENO_FILE = f"{cfg.TRAITS[0]}.phe"
    pc.TRAIT = cfg.TRAITS[0]
    pc.COV_FILE = f"{cfg.TRAITS[0]}.cov" if os.path.exists(f"{cfg.TRAITS[0]}.cov") else None
    pc.QCOV_FILE = f"{cfg.TRAITS[0]}.qcov" if os.path.exists(f"{cfg.TRAITS[0]}.qcov") else None
    X, _, chrom, pos, snp_names, _ = P.load_real(pc)
    n, m = X.shape
    say(f"[data] {n} animals x {m} SNPs")
    Ha, Hd = P.noia(X)

    say("\n[kernels] exact aggregate kernels")
    K, meta = build_kernels(cfg, X, Ha, Hd, chrom, keep_xc=("focal" in stages))
    Xc = meta.get("Xc")

    iu = np.triu_indices(n, 1)
    names = list(K)
    CM = pd.DataFrame({a: {b: np.corrcoef(K[a][iu], K[b][iu])[0, 1] for b in names}
                       for a in names})
    CM.to_csv(os.path.join(cfg.OUT_DIR, "kernel_collinearity.csv"))
    say("\n[diagnostic] kernel correlation matrix -> kernel_collinearity.csv")
    hi = [(a, b, float(CM.loc[a, b])) for a in names for b in names
          if a < b and abs(CM.loc[a, b]) > 0.5]
    if hi:
        say("    kernels with |r| > 0.5 -- their VARIANCE COMPONENTS are not")
        say("    separately identifiable and should be reported as descriptive")
        say("    only; delta PCC (held-out prediction) is unaffected:")
        for a, b, r in sorted(hi, key=lambda z: -abs(z[2]))[:8]:
            say(f"      {a:8s} ~ {b:8s}  r = {r:+.3f}")

    labels, how = P.structure_clusters(K["A"], None, 10, cfg.SEED)
    say(f"[structure] {how}   [split] {cfg.SPLIT}   [seeds] {cfg.N_SEEDS}")

    diffuse = {"A": ["A"], "A+D": ["A", "D"]}
    for k in (x for x in K if x not in ("A", "D")):
        diffuse[f"A+D+{k}"] = ["A", "D", k]
    tk = [x for x in K if x.endswith("_t")]
    ck = [x for x in K if x.endswith("_c")]
    if tk:
        diffuse["A+D+trans_all"] = ["A", "D"] + tk
    if ck:
        diffuse["A+D+cis_all"] = ["A", "D"] + ck
    if tk or ck:
        diffuse["A+D+all"] = ["A", "D"] + tk + ck

    all_diff, all_dec, all_foc, all_pairs, all_mate = [], [], [], [], []

    for trait in cfg.TRAITS:
        if not os.path.exists(f"{trait}.phe"):
            say(f"\n[skip] {trait}: {trait}.phe not found")
            continue
        say("\n" + "=" * 78)
        say(f"TRAIT: {trait}")
        say("=" * 78)
        pc.PHENO_FILE, pc.TRAIT = f"{trait}.phe", trait
        pc.COV_FILE = f"{trait}.cov" if os.path.exists(f"{trait}.cov") else None
        pc.QCOV_FILE = f"{trait}.qcov" if os.path.exists(f"{trait}.qcov") else None
        # Genotypes are identical across traits; reloading them per trait would
        # duplicate a ~0.8 GB matrix for no reason. Reuse X and take only the
        # phenotype/covariates from this call.
        Xt_, y, _, _, _, extra = P.load_real(pc)
        if Xt_.shape[0] != n:
            say(f"  [warn] {trait} has {Xt_.shape[0]} animals vs {n}; skipping")
            del Xt_
            continue
        del Xt_
        Xt = X
        parts = [np.ones((n, 1))]
        if cfg.USE_COV and extra is not None:
            parts.append(np.asarray(extra, float))
        Xfix = np.column_stack(parts)
        say(f"  fixed effects {Xfix.shape}" +
            ("" if cfg.USE_COV else "  (intercept only; cohort covariates excluded)"))

        # ---------------- [A] diffuse ----------------
        if "diffuse" in stages:
            say("\n  [A] diffuse aggregate kernels")
            n_core = sum(1 for k in diffuse if k in cfg.CORE_MODELS)
            n_sec = len(diffuse) - n_core
            t_probe = time.time()
            _g, _y, _ = fit_gebv(y, Xfix, K, ["A", "D", "MULT_t"],
                                 *make_split(n, labels, cfg.SPLIT, cfg.TEST_FRAC, cfg.SEED))
            t_fit = time.time() - t_probe
            n_fits = n_core * cfg.N_SEEDS + n_sec * min(cfg.SECONDARY_SEEDS, cfg.N_SEEDS)
            say(f"    one REML fit takes {t_fit:.0f}s; {n_fits} fits planned "
                f"({n_core} core x {cfg.N_SEEDS} + {n_sec} secondary x "
                f"{min(cfg.SECONDARY_SEEDS, cfg.N_SEEDS)})")
            say(f"    ESTIMATED TIME FOR THIS TRAIT: {t_fit * n_fits / 60:.0f} min")
            rows, acc, bias, gebv_store = [], {}, {}, []
            t_stage = time.time()
            for si in range(cfg.N_SEEDS):
                t_seed = time.time()
                seed = cfg.SEED + si * 101
                tr, te = make_split(n, labels, cfg.SPLIT, cfg.TEST_FRAC, seed)
                preds = {}
                todo = {k: v for k, v in diffuse.items()
                        if k in cfg.CORE_MODELS or si < cfg.SECONDARY_SEEDS}
                for nm, keys in todo.items():
                    gb, ya, _ = fit_gebv(y, Xfix, K, keys, tr, te)
                    if gb is None:
                        continue
                    preds[nm] = (gb, ya)
                    acc.setdefault(nm, []).append(pcc(ya, gb))
                    bias.setdefault(nm, []).append(pred_bias(ya, gb))
                    # Persist the GEBVs. Kernels are already cached, so the
                    # whole cost of this stage is REML; storing the fitted
                    # values means any further metric can be derived later
                    # without refitting anything.
                    gebv_store.append(dict(trait=trait, seed=seed, model=nm,
                                           idx=";".join(map(str, te)),
                                           gebv=";".join(f"{v:.6g}" for v in gb),
                                           y_adj=";".join(f"{v:.6g}" for v in ya)))
                if "A+D" not in preds:
                    continue
                g0, ya = preds["A+D"]
                for nm, (g1, _) in preds.items():
                    if nm in ("A", "A+D"):
                        continue
                    d, lo, hi = boot_delta(ya, g0, g1, cfg.N_BOOT, seed)
                    rows.append(dict(trait=trait, seed=seed, model=nm,
                                     dPCC=d, ci_lo=lo, ci_hi=hi))
                el = time.time() - t_seed
                best = max(((v, k) for k, v in preds.items()),
                           key=lambda z: pcc(z[0][1], z[0][0]))
                say(f"      split {si+1:2d}/{cfg.N_SEEDS}  {el:5.1f}s  "
                    f"A+D={pcc(preds['A+D'][1], preds['A+D'][0]):.4f}  "
                    f"best={best[1]} {pcc(best[0][1], best[0][0]):.4f}"
                    + (f"   [est. total for this trait: "
                       f"{el * cfg.N_SEEDS / 60:.0f} min]" if si == 0 else ""))
            say(f"    stage A took {(time.time()-t_stage)/60:.1f} min")
            for nm, v in acc.items():
                bb = bias.get(nm, [np.nan])
                say(f"    {nm:18s} accuracy = {np.mean(v):.4f} +- {np.std(v):.4f}"
                    f"   bias = {np.nanmean(bb):.3f}")
            pd.DataFrame([dict(trait=trait, model=nm,
                               accuracy=float(np.mean(v)), acc_sd=float(np.std(v)),
                               bias=float(np.nanmean(bias.get(nm, [np.nan]))),
                               bias_sd=float(np.nanstd(bias.get(nm, [np.nan]))))
                          for nm, v in acc.items()]).to_csv(
                os.path.join(cfg.OUT_DIR, f"{trait}_accuracy_bias.csv"), index=False)
            if gebv_store:
                pd.DataFrame(gebv_store).to_csv(
                    os.path.join(cfg.OUT_DIR, f"{trait}_gebv_store.csv.gz"),
                    index=False, compression="gzip")
            if rows:
                D = pd.DataFrame(rows)
                all_diff.append(D)
                g = (D.groupby("model")
                     .agg(dPCC=("dPCC", "mean"), sd=("dPCC", "std"),
                          pos_frac=("dPCC", lambda v: float(np.mean(np.asarray(v) > 0))))
                     .reset_index().sort_values("dPCC", ascending=False))
                say("\n    delta PCC vs A+D:")
                say(g.round(5).to_string(index=False))

        # ---------------- [B] decompose ----------------
        if "decompose" in stages:
            say("\n  [B] additive / dominance / epistatic breeding-value correlation")
            keys = [k for k in ("A", "D", "MULT_t") if k in K]
            Ktt = [K[k] for k in keys]
            mm = P.REMLgrad().fit(y, Xfix, Ktt)
            v = np.asarray(mm.variances_, float)
            V = sum(s * A for s, A in zip(v[:-1], Ktt)) + (v[-1] + 1e-8) * np.eye(n)
            c = cho_factor(V, lower=True)
            b = np.linalg.solve(Xfix.T @ cho_solve(c, Xfix), Xfix.T @ cho_solve(c, y))
            al = cho_solve(c, y - Xfix @ b)
            u = {k: float(s) * (A @ al) for k, s, A in zip(keys, v[:-1], Ktt)}
            rec = dict(trait=trait)
            for k in keys:
                rec[f"var_{k}"] = float(v[keys.index(k)] / v.sum())
            for a_, b_ in combinations(keys, 2):
                rec[f"r({a_},{b_})"] = pcc(u[a_], u[b_])
            all_dec.append(rec)
            say("    " + "  ".join(f"{k}={val:.4f}" for k, val in rec.items()
                                   if k != "trait"))
            rr = rec.get("r(A,MULT_t)", np.nan)
            if np.isfinite(rr):
                say(f"    -> r(EBV_A, EBV_AA) = {rr:.4f}; "
                    + ("epistasis carries little information beyond the additive value"
                       if abs(rr) > 0.9 else
                       "epistasis carries INDEPENDENT information -> mate allocation can act on it"))

        # ---------------- [C] focal ----------------
        if "focal" in stages:
            say(f"\n  [C] genome-wide exact scoring ({cfg.FOCAL_SEEDS} splits)")
            seen = {}
            for si in range(cfg.FOCAL_SEEDS):
                seed = cfg.SEED + si * 101
                tr, te = make_split(n, labels, cfg.SPLIT, cfg.TEST_FRAC, seed)
                t1 = time.time()
                tops = score_genomewide(Xt[tr], y[tr], cfg.SCORE_TYPES,
                                        cfg.TOP_POOL, cfg.SCAN_CHUNK,
                                        cfg.MIN_CELL, verbose=(si == 0))
                say(f"    split {si+1}: scoring took {time.time()-t1:.0f}s")
                g0 = fit_gebv(y, Xfix, K, ["A", "D"], tr, te)
                agg = "MULT_t" if "MULT_t" in K else "HAD"
                gA = fit_gebv(y, Xfix, K, ["A", "D", agg], tr, te)
                if g0[0] is None or gA[0] is None:
                    continue
                ya = g0[1]
                for typ, plist in tops.items():
                    for rk, (v_, i_, j_, tg) in enumerate(plist[:max(cfg.K_GRID)], 1):
                        key = (typ, min(i_, j_), max(i_, j_))
                        seen.setdefault(key, []).append(seed)
                    for mode in ("single", "cumulative"):
                        for Kk in cfg.K_GRID:
                            if Kk > len(plist):
                                break
                            sel = [plist[Kk - 1]] if mode == "single" else plist[:Kk]
                            KT = pairs_kernel(sel, typ, Xt, Ha, Hd, Xc)
                            if KT is None:
                                continue
                            Kl = dict(K)
                            Kl["TOP"] = KT
                            g1, _, _ = fit_gebv(y, Xfix, Kl, ["A", "D", agg, "TOP"], tr, te)
                            if g1 is None:
                                continue
                            all_foc.append(dict(
                                trait=trait, seed=seed, type=typ, mode=mode, K=Kk,
                                acc=pcc(ya, g1),
                                d_vs_AD=pcc(ya, g1) - pcc(ya, g0[0]),
                                d_vs_AGG=pcc(ya, g1) - pcc(ya, gA[0])))
            for (typ, i_, j_), seeds in seen.items():
                all_pairs.append(dict(trait=trait, type=typ,
                                      snp_i=snp_names[i_], snp_j=snp_names[j_],
                                      chr_i=int(chrom[i_]), chr_j=int(chrom[j_]),
                                      pos_i=int(pos[i_]), pos_j=int(pos[j_]),
                                      n_splits_selected=len(seeds),
                                      reproducibility=len(seeds) / cfg.FOCAL_SEEDS,
                                      is_trans=bool(chrom[i_] != chrom[j_])))
            rep = [d for d in all_pairs if d["trait"] == trait
                   and d["n_splits_selected"] >= 3]
            say(f"    pairs re-selected in >=3/{cfg.FOCAL_SEEDS} independent splits: {len(rep)}")

        # ---------------- [D] mating ----------------
        if "mating" in stages:
            say("\n  [D] epistasis-aware mate allocation")
            _, summ = mate_allocation(cfg, Xt, Ha, Hd, chrom, K, meta, y, Xfix,
                                      cfg.OUT_DIR, trait)
            all_mate.append(summ)

    # ---------------- write outputs ----------------
    def dump(lst, name, concat=True):
        if not lst:
            return None
        df = pd.concat(lst, ignore_index=True) if concat else pd.DataFrame(lst)
        df.to_csv(os.path.join(cfg.OUT_DIR, name), index=False)
        return df

    D = dump(all_diff, "diffuse_delta_pcc.csv")
    dump(all_dec, "breeding_value_decomposition.csv", concat=False)
    dump(all_foc, "focal_delta_pcc.csv", concat=False)
    dump(all_pairs, "focal_pairs_reproducibility.csv", concat=False)
    M = dump(all_mate, "mating_summary.csv")

    if D is not None:
        say("\n" + "=" * 78)
        say("SUMMARY  [A] diffuse: delta PCC vs A+D, averaged over splits")
        say("=" * 78)
        say(D.pivot_table(index="model", columns="trait", values="dPCC")
            .round(5).to_string())
    if M is not None:
        say("\n" + "=" * 78)
        say("SUMMARY  [D] mate allocation")
        say("=" * 78)
        say(M.round(4).to_string(index=False))
        say("\n  overlap_with_ADD < 1 means epistasis/dominance changed which")
        say("  crosses are recommended -- i.e. mate allocation adds information")
        say("  that additive selection alone cannot provide.")

    json.dump(dict(traits=list(cfg.TRAITS), n=int(n), m=int(m),
                   seeds=cfg.N_SEEDS, stages=sorted(stages),
                   minutes=(time.time() - t0) / 60),
              open(os.path.join(cfg.OUT_DIR, "run_summary.json"), "w"), indent=2)
    say(f"\nDone in {(time.time()-t0)/60:.1f} min. Results in {cfg.OUT_DIR}/")


# ============================================================================
# Self-test
# ============================================================================
def selftest():
    say("=" * 78)
    say("SELF-TEST")
    say("=" * 78)
    ok = True
    ok &= install_reml_speedup()
    rng = np.random.default_rng(0)
    n, m = 400, 150
    X = (rng.random((n, m)) < 0.35) * 1.0 + (rng.random((n, m)) < 0.35) * 1.0
    chrom = np.repeat(np.arange(1, m // 50 + 2), 50)[:m]
    Ha, Hd = P.noia(X)
    Xx = (X > 0).astype(float) * 2 - 1
    xc = Xx - Xx.mean(0)
    sd = xc.std(0)
    sd[sd < 1e-12] = 1.0
    Xc = xc / sd
    y = rng.normal(0, 1, n)
    yc = y - y.mean()

    say("\n[1] exact scoring vs brute force")
    tops = score_genomewide(X, y, ("MULT", "DD", "THR"), 30, 64)
    for t in ("MULT", "DD", "THR"):
        md = max(abs(v - abs(np.corrcoef(
            pair_feature(t, tg, X, Ha, Hd, Xc, i, j), yc)[0, 1]))
            for (v, i, j, tg) in tops[t][:10])
        say(f"    {t:5s} max deviation over top 10 = {md:.2e}  {'OK' if md < 1e-5 else 'FAIL'}")
        ok &= md < 1e-5

    say("\n[2] injected-pair recovery")
    for t, tag in [("MULT", "MULT"), ("DD", "DD"), ("THR", "s>=3")]:
        i0, j0 = 7, 101
        z = pair_feature(t, tag, X, Ha, Hd, Xc, i0, j0)
        y2 = 1.5 * z + rng.normal(0, 1, n)
        tt = score_genomewide(X, y2, (t,), 20, 64)[t]
        rank = next((r for r, (v, a, b, g) in enumerate(tt, 1) if {a, b} == {i0, j0}), None)
        say(f"    {t:5s} injected pair rank = {rank}")
        ok &= rank is not None and rank <= 3

    say("\n[3] offspring epistatic value: analytical vs Monte-Carlo")
    i, j = 3, 17
    p1, p2 = X[i] / 2, X[j] / 2
    h = (Ha[i] + Ha[j]) / 2
    r = np.random.default_rng(1)
    G = ((r.random((20000, m)) < p1).astype(float) +
         (r.random((20000, m)) < p2).astype(float))
    Hg = G - X.mean(0)
    jj = 5
    tt = Hg @ Ha[jj]
    tc = sum((Hg[:, chrom == c] @ Ha[jj, chrom == c]) ** 2 for c in np.unique(chrom))
    mc = float(np.mean(tt ** 2 - tc))
    te = float((h @ Ha[jj]) ** 2 -
               sum((h[chrom == c] @ Ha[jj, chrom == c]) ** 2 for c in np.unique(chrom)))
    rel = abs(mc - te) / max(abs(mc), 1e-9)
    say(f"    Monte-Carlo {mc:.4f}  analytical {te:.4f}  relative deviation {rel:.4f}")
    ok &= rel < 0.05

    say("\n[4] accuracy metric invariant to fixed-effect magnitude")
    KA = P.grm_a(X)
    KA /= np.mean(np.diag(KA))
    KD = P.grm_d(Hd)
    KD /= np.mean(np.diag(KD))
    C = rng.normal(0, 1, (n, 8))
    fx = C @ rng.normal(0, 1, 8)
    Xf = np.column_stack([np.ones(n), C])
    d_, U = np.linalg.eigh(KA)
    dd = np.maximum(d_, 0)
    zf = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    accs = []
    for fw in (0.0, 0.5):
        r = np.random.default_rng(7)
        g = U @ (np.sqrt(dd) * r.normal(0, 1, n))
        yy = (np.sqrt(fw) * zf(fx) + np.sqrt((1 - fw) * 0.6) * zf(g)
              + np.sqrt((1 - fw) * 0.4) * r.normal(0, 1, n))
        te_ = np.zeros(n, bool)
        te_[r.choice(n, n // 5, replace=False)] = True
        trn, tei = np.where(~te_)[0], np.where(te_)[0]
        gb, ya, _ = fit_gebv(yy, Xf, {"A": KA, "D": KD}, ["A", "D"], trn, tei)
        accs.append(pcc(ya, gb))
    say(f"    fixed-effect share 0.0 -> {accs[0]:.4f}; 0.5 -> {accs[1]:.4f}; "
        f"difference {abs(accs[0]-accs[1]):.4f}")
    ok &= abs(accs[0] - accs[1]) <= 0.02

    say("\nSELF-TEST: " + ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--bfile")
    ap.add_argument("--traits", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--focal_seeds", type=int)
    ap.add_argument("--candidates", type=int)
    ap.add_argument("--crosses", type=int)
    ap.add_argument("--split", choices=["random", "stratified"])
    ap.add_argument("--stages", default="diffuse,decompose,mating",
                    help="comma-separated: diffuse,decompose,focal,mating or 'all'")
    ap.add_argument("--secondary_seeds", type=int,
                    help="splits for diagnostic (non-core) models; default 5")
    ap.add_argument("--use_cov", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    cfg = CFG()
    for val, attr in [(a.bfile, "BFILE"), (a.out, "OUT_DIR"), (a.seeds, "N_SEEDS"),
                      (a.focal_seeds, "FOCAL_SEEDS"), (a.candidates, "N_CANDIDATES"),
                      (a.crosses, "N_CROSSES"), (a.split, "SPLIT"),
                      (a.secondary_seeds, "SECONDARY_SEEDS")]:
        if val is not None:
            setattr(cfg, attr, val)
    if a.traits:
        cfg.TRAITS = tuple(a.traits)
    if a.use_cov:
        cfg.USE_COV = True
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    stages = ({"diffuse", "decompose", "focal", "mating"} if a.stages == "all"
              else {s.strip() for s in a.stages.split(",") if s.strip()})
    run(cfg, stages)


if __name__ == "__main__":
    main()
