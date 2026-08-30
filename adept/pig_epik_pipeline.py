#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical core of the epistasis discovery pipeline (species-agnostic)
================================================================================
This module is the numerical foundation shared by the whole repository: PLINK
binary reading, NOIA coding, exact construction of additive, dominance and
epistatic kernels, REML with an analytical gradient, structure clustering and
stratified splitting, and pairwise interaction scoring. Other modules reuse it
as `import pig_epik_pipeline as P`.

Three methodological conventions run through the module.

  1  The criterion is the gain in predictive correlation on an independent test
     set. An aggregate epistatic kernel is a PREDICTION component only: its
     value is the improvement from A+D to A+D+AA. At these sample sizes a
     variance fraction cannot be compared with zero -- on purely additive
     simulated data a spurious epistatic fraction reaches 0.687, because the
     REML likelihood itself peaks there. Variance fractions are printed as
     diagnostics and never used to decide whether epistasis is present.

  2  Individuals used to choose loci or tune a threshold must not be used to
     validate the gain. The 20% test set is held out completely; pair scoring
     runs inside the 80% training set only, so nothing leaks.

  3  The statistical core exists once in the project: REML (analytical gradient
     + L-BFGS), noia / grm_a / grm_d / hadamard / pca_from_grm,
     interaction_F_conditioned with its genotype-cell filter, and main-effect
     conditioning through quick_additive_gwas / pick_major_effect_snps. The AA
     (additive x additive) and threshold tests use that same framework, so
     their NOIA coding agrees with AD / DD / MULT / XOR.

--------------------------------------------------------------------------------
PIPELINE STRUCTURE

  [Part 0] load data + ADMIXTURE structure  ->  80:20 stratified split
              train 80% (OUTER_TRAIN)           test 20% (OUTER_TEST, frozen)

  [Part 1] 5-fold CV genomic prediction inside the training set (multi-kernel
           GBLUP REML) for A / A+D / A+D+AA / A+D+AA+DD+AD
              -> per-fold PCC, mean +/- SD, and the standardised epistatic
                 gain dPCC(A+D+AA - A+D)
           also: fit on the whole training set, predict the frozen 20%, and
           report dPCC with a paired-bootstrap 95% CI

  [Part 2] localisation kernels cut the search space (training set only)
              chromosome-level kernel -> cross-chromosome pairs (trans)
              LD-level kernel         -> nearby pairs (cis); the window follows
                                         the --indep-pairwise pruning scale

  [Part 3] six interaction tests (AA / AD / DD / MULT / threshold / XOR) score
           every pair in both search spaces, and every result is written out.
           No threshold or significance filter is applied here; filtering is
           the job of the downstream scripts.

--------------------------------------------------------------------------------
ON THE LD SCALE
  The input genotypes are already pruned with `--indep-pairwise 500 50 0.1`, so
  any two retained neighbouring SNPs have r2 < 0.1 and are close to independent.
  The near (cis) search space is therefore defined operationally by a physical
  window, NEAR_WINDOW_KB, chosen by the caller and stated in the methods.
  Physical distance alone cannot decide whether a signal is an LD artefact --
  a real interaction between two adjacent small genes would be discarded with
  it -- so near pairs are scanned like any other and their status is settled by
  the later prediction check, not by a distance rule applied in advance.

--------------------------------------------------------------------------------
INPUT
  BFILE(.bed/.bim/.fam)   PLINK binary, already QC-filtered and LD-pruned
  PHENO_FILE              GCTA format: FID IID value
  COV_FILE / QCOV_FILE    optional GCTA covariates (batch, recording weight...)

OUTPUT DIRECTORY  pig_epik_out/
  split_assignment.csv        ADMIXTURE cluster and train/test status per
                              individual
  structure_info.json         structure decision (K, CV error, cluster shares,
                              split seed)
  prediction_cv.csv           per-fold and mean +/- SD PCC of the four models
  prediction_cv_delta.csv     per-fold dPCC (epistatic gain) and mean +/- SD
  prediction_holdout.csv      frozen test set PCC, dPCC, paired-bootstrap 95% CI
  variance_components.csv     variance fractions per model (diagnostic only;
                              never to be read as evidence for or against
                              epistasis)
  conditioned_major_snps.csv  large-effect main loci removed as fixed effects
  candidate_pool.csv          marginal screening pool (search space for the
                              AA and threshold tracks)
  epistasis_scores_all.csv    every pair x every one of the six tests, unfiltered
  run_summary.json            full configuration and headline numbers of a run

Dependencies: numpy, scipy, pandas. joblib is optional (parallel scoring);
plink and admixture are optional and fall back to PCA clustering if absent.

Self-test:  python3 pig_epik_pipeline.py --selftest
            Simulates structured data with six kinds of true interaction
            injected, runs the whole pipeline and prints PASS/FAIL consistency
            checks.
"""
import os, sys, json, re, shutil, subprocess, time, argparse, warnings
import pickle, hashlib
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve, LinAlgError
from scipy.optimize import minimize
from scipy.stats import chi2, pearsonr, spearmanr
warnings.filterwarnings("ignore")

from multiprocessing import cpu_count as _host_cpu_count
from contextlib import contextmanager


def cpu_count():
    """
    Inside a container multiprocessing.cpu_count() returns the number of host
    cores (up to 256) while the process is limited to its quota. Sizing thread
    pools from that number oversubscribes several-fold and runs slower than
    serial execution. CPU affinity, then the cgroup v2/v1 quota, then the host
    count are read in turn and the smallest is used.
    """
    cands = []
    try:
        cands.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    for quota_f, period_f in [("/sys/fs/cgroup/cpu.max", None),
                              ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                               "/sys/fs/cgroup/cpu/cpu.cfs_period_us")]:
        try:
            if period_f is None:
                q, p = open(quota_f).read().split()
                if q != "max":
                    cands.append(max(1, int(float(q) / float(p))))
            else:
                q = int(open(quota_f).read()); p = int(open(period_f).read())
                if q > 0:
                    cands.append(max(1, int(q / p)))
        except (OSError, ValueError):
            pass
    try:
        cands.append(_host_cpu_count())
    except Exception:
        cands.append(1)
    return max(1, min(cands))


def avail_mem_gb():
    """Container memory limit from cgroup where available, host memory otherwise."""
    cands = []
    for cur_f, lim_f in [("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                         ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
                          "/sys/fs/cgroup/memory/memory.limit_in_bytes")]:
        try:
            lim = open(lim_f).read().strip()
            if lim != "max":
                lim = int(lim)
                if lim < (1 << 60):
                    cands.append((lim - int(open(cur_f).read())) / 1e9)
        except (OSError, ValueError):
            pass
    try:
        cands.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9)
    except (ValueError, OSError):
        pass
    return max(0.5, min(cands)) if cands else 8.0
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False


@contextmanager
def par_ctx(inner_threads=1):
    """
    Limit the number of BLAS threads PER WORKER PROCESS, across joblib versions.

    inner_max_num_threads is an argument of parallel_config / parallel_backend,
    NOT of Parallel(); passing it to Parallel() raises TypeError. Without the
    limit, N worker processes each running M BLAS threads contend for N*M
    threads, which on a 48-core machine is slower than running serially.
    """
    if not _HAS_JOBLIB:
        yield; return
    try:                       # joblib >= 1.3
        from joblib import parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=inner_threads):
            yield
        return
    except Exception:
        pass
    try:                       # older joblib
        from joblib import parallel_backend
        with parallel_backend("loky", inner_max_num_threads=inner_threads):
            yield
        return
    except Exception:
        pass
    yield                      # neither available: run unrestricted


# ============================== Configuration ==============================
class CFG:
    # --- input ---
    BFILE      = "genotypes"                    # .bed/.bim/.fam prefix; set via --bfile
    PHENO_FILE = "bf.phe"                       # GCTA: FID IID value
    COV_FILE   = None                           # optional GCTA .cov  (categorical)
    QCOV_FILE  = None                           # optional GCTA .qcov (continuous)
    TRAIT      = "BF"

    OUT_DIR    = "pig_epik_out"
    SEED       = 2024

    # --- Part 0: structure and splitting ---
    N_PCS            = 5           # structure PCs entering the fixed effects
    TEST_FRAC        = 0.20        # share held out as the independent test set
    ADMIX_K_RANGE    = range(1, 4) # candidate K for ADMIXTURE (a closed herd
    PLINK_BIN        = "plink"
    ADMIX_BIN        = "admixture"
    FALLBACK_MAX_K   = 4           # largest K for PCA+KMeans when ADMIXTURE is absent

    # --- Part 1: prediction ---
    N_CV_FOLDS       = 5
    N_BOOT           = 2000        # paired bootstrap replicates
    REML_RESTARTS    = 2
    REML_MAXITER     = 100

    # --- Part 2: search space ---
    POOL_SIZE        = 1000        # track-1 marginal screening pool (AA/AD/MULT)
    MAX_MAJOR_SNPS   = 10          # at most this many large main-effect loci are
    NEAR_WINDOW_KB   = 1000        # physical window (kb) defining the near (cis)
    MAX_NEAR_PER_ANCHOR = 200      # neighbours per pool anchor (caps the near-pair
    XOR_N_RANDOM_PAIRS  = 200000   # randomly sampled pairs for the XOR track

    # --- track 2: no marginal pool; genome-wide cover for DD / threshold / XOR ---
    CIS_WINDOW_MB        = 1.0      # cis physical window (Mb) for DD and threshold
    MAX_CIS_NEIGHBORS    = 20       # neighbours per SNP inside the cis window; after
                                    # pruning, distant pairs are close to independent
    XOR_MIN_MAF          = 0.25     # minimum MAF for the XOR sampling pool
    XOR_MIN_HET          = 0.10     # minimum heterozygote frequency for that pool
    XOR_MAX_PAIRS        = 2000000  # cap on cross-chromosome XOR pairs scored

    # --- Part 1.5: multi-resolution slice kernels and the gate ---
    KERNEL_THRESH_K      = 2        # count>=k used by the threshold kernel (pair
    KERNEL_TRANS_SAMPLE  = 200000   # Monte-Carlo sample size for the trans threshold
    GATE_REML_MAXITER    = 40       # REML iteration cap during gating (screening only)
    FDR_Q                = 0.20     # BH-FDR level used by the gate
    STABILITY_MAX_CV     = 1.0      # gate stability limit: CV of the per-fold dPCC

    # --- control for phantom cis epistasis ---
    HAP_CTRL_PCS         = 3        # local-window haplotype PCs
    HAP_CTRL_TOP_N       = 200      # how many cis hits get the haplotype control
    HAP_CTRL_KEEP_FRAC   = 0.5      # fraction of F that must survive the control

    # --- evaluation metrics ---
    TOP_FRAC             = 0.10     # selected proportion for the realised-gain metric
    OVERLAP_FRAC         = 0.20     # overlap of predicted and observed top fraction

    # --- Part 3: scoring ---
    MIN_CELL_COUNT_EFF   = 20      # genotype-cell safety valve; not a discovery filter
    THRESHOLD_K_RANGE = (1, 2, 3, 4)   # count>=k scanned by the threshold test
    N_JOBS            = 1              # scoring worker processes (1 = serial)

    JITTER = 1e-6

    # --- checkpointing ---
    RESUME          = True      # reuse completed stages found in CKPT_DIR
    CKPT_DIR        = None      # None => OUT_DIR/_ckpt
    SHARED_CKPT_DIR = "shared_ckpt"   # GRM / structure / slice kernels are trait-
    PART1_ONLY      = False     # stop after Part 1 (fast trait screen); no slice kernels
    RESPECT_GATE    = True      # if no slice kernel passes the gate, skip pair scanning
    USE_ADMIXTURE   = True      # False: use PCA+KMeans (seconds) instead of ADMIXTURE
    ADMIX_THREADS   = 8         # ADMIXTURE -j threads

    # --- parallelism and memory ---
    GATE_N_JOBS     = 1         # gate worker processes (1 = serial)
    MEM_FRACTION    = 0.5       # largest share of free memory the scoring pool may use


# =====================================================================
#  Statistical primitives: NOIA coding, GRM, PCA
# =====================================================================

def noia(X):
    """NOIA additive (Ha) and dominance (Hd) coding."""
    pAA = np.mean(X == 0, 0); pAa = np.mean(X == 1, 0); paa = np.mean(X == 2, 0)
    mu = pAa + 2 * paa; Ha = X - mu
    v = pAA * mu**2 + pAa * (1 - mu)**2 + paa * (2 - mu)**2
    mono = v < 1e-12
    H = (X == 1).astype(float)
    b = np.where(mono, 0, (pAa * (1 - mu)) / np.where(mono, 1, v))
    Hd = H - pAa - b * Ha
    Hd[:, mono] = 0.
    return Ha, Hd

def grm_a(X):
    p = X.mean(0) / 2; Z = X - 2 * p
    return (Z @ Z.T) / (2 * np.sum(p * (1 - p)) + 1e-12)

def grm_d(Hd):
    return (Hd @ Hd.T) / (np.sum(np.var(Hd, 0)) + 1e-12)

def hadamard(K1, K2):
    K = K1 * K2
    return K / (np.mean(np.diag(K)) + 1e-12)

def pca_from_grm(K, n):
    if n <= 0:
        return np.zeros((K.shape[0], 0))
    ev, evec = np.linalg.eigh(K)
    o = np.argsort(ev)[::-1][:n]
    return evec[:, o] * np.sqrt(np.maximum(ev[o], 0))


# ---------- Multi-kernel REML (analytical gradient + L-BFGS) ----------
class REMLgrad:
    """Multi-kernel REML with an analytical gradient, verified against finite differences."""
    def __init__(self, jitter=CFG.JITTER):
        self.jitter = jitter

    def _obj_grad(self, theta, y, X, Ks):
        n = len(y)
        try:
            v = np.exp(theta)
            V = v[-1] * np.eye(n) + self.jitter * np.eye(n)
            for vi, K in zip(v[:-1], Ks):
                V = V + vi * K
            L = cholesky(V, lower=True)
            Vi = cho_solve((L, True), np.eye(n))
            ViX = Vi @ X; XtViX = X.T @ ViX
            Li = cholesky(XtViX, lower=True)
            P = Vi - ViX @ cho_solve((Li, True), np.eye(X.shape[1])) @ ViX.T
            Py = P @ y
            nll = 0.5 * (2 * np.sum(np.log(np.diag(L)))
                         + 2 * np.sum(np.log(np.diag(Li))) + float(y @ Py))
        except LinAlgError:
            return 1e10, np.zeros_like(theta)
        g = np.empty(len(theta))
        for i, K in enumerate(list(Ks) + [None]):
            if K is None:
                tr = np.trace(P); quad = float(Py @ Py)
            else:
                tr = np.trace(P @ K); quad = float(Py @ (K @ Py))
            g[i] = 0.5 * (tr - quad) * v[i]
        return nll, g

    def fit(self, y, X, Ks, n_restarts=CFG.REML_RESTARTS, maxiter=CFG.REML_MAXITER):
        best = None; vy = float(np.var(y)); nv = len(Ks) + 1
        rng = np.random.RandomState(0)
        for r in range(n_restarts):
            sc = np.ones(nv) if r == 0 else rng.uniform(0.3, 2.5, nv)
            th0 = np.log(np.maximum(vy / nv * sc, 1e-6))
            res = minimize(self._obj_grad, th0, args=(y, X, Ks), jac=True,
                           method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-9})
            if best is None or res.fun < best.fun:
                best = res
        self.variances_ = np.exp(best.x); self.loglik_ = -best.fun
        return self

    def predict_new(self, y, X, Ks_tt, Xnew, Ks_new_t):
        """GBLUP prediction for new individuals: u = sum_k v_k K_{new,train} V^-1 (y - Xb)."""
        n = len(y); v = self.variances_
        V = v[-1] * np.eye(n) + self.jitter * np.eye(n)
        for vi, K in zip(v[:-1], Ks_tt):
            V = V + vi * K
        L = cholesky(V, lower=True)
        b = np.linalg.solve(X.T @ cho_solve((L, True), X), X.T @ cho_solve((L, True), y))
        Vr = cho_solve((L, True), y - X @ b)
        u = np.zeros(Ks_new_t[0].shape[0])
        for vi, Knt in zip(v[:-1], Ks_new_t):
            u += vi * (Knt @ Vr)
        return Xnew @ b + u


def lrt_boundary(ll_full, ll_red, df=1):
    """Boundary LRT: a 0.5*chi2_0 + 0.5*chi2_df mixture. Diagnostic use only."""
    stat = max(2.0 * (ll_full - ll_red), 0.0)
    if stat <= 0:
        return 0.0, 1.0
    return stat, 0.5 * float(chi2.sf(stat, df))


# ---------- Conditioning on large main-effect loci ----------
def quick_additive_gwas(X, y, Xdesign_base):
    """Fast additive GWAS inside the training set. Returns -log10(p) and beta per SNP."""
    n, m = X.shape
    Q, _ = np.linalg.qr(Xdesign_base)
    y_r = y - Q @ (Q.T @ y)
    Xc = X - Q @ (Q.T @ X)
    sxx = np.einsum('ij,ij->j', Xc, Xc) + 1e-12
    sxy = Xc.T @ y_r
    beta = sxy / sxx
    dfe = n - Xdesign_base.shape[1] - 1
    rss = float(y_r @ y_r) - beta * sxy
    se2 = np.maximum(rss, 1e-12) / dfe / sxx
    t2 = beta ** 2 / np.maximum(se2, 1e-30)
    logp = -np.log10(np.maximum(chi2.sf(t2, df=1), 1e-300))
    return logp, beta

def pick_major_effect_snps(logp, snp_names, X, n_max=CFG.MAX_MAJOR_SNPS,
                           thresh=None, ld_window_r2=0.5):
    """Pick large main-effect loci to condition on (Bonferroni threshold, greedy decorrelation)."""
    m = len(logp)
    if thresh is None:
        thresh = -np.log10(0.05 / max(m, 1))
    order = np.argsort(-logp)
    picked = []
    for k in order:
        if logp[k] < thresh or len(picked) >= n_max:
            break
        ok = True
        for q in picked:
            r = np.corrcoef(X[:, k], X[:, q])[0, 1]
            if np.isfinite(r) and r ** 2 > ld_window_r2:
                ok = False; break
        if ok:
            picked.append(k)
    return np.array(picked, dtype=int), thresh


# ---------- Interaction testing core ----------
MIN_CELL_COUNT = 20   # minimum individuals per relevant genotype cell. This
                      # removes degenerate interactions; it is not a discovery
                      # threshold

def _cells_ok(g1, g2, mode, min_count=MIN_CELL_COUNT):
    """Whether a pair has enough individuals to support this type of interaction test."""
    if mode == "DD":
        h1 = int(np.sum(g1 == 1)); h2 = int(np.sum(g2 == 1))
        both = int(np.sum((g1 == 1) & (g2 == 1)))
        return (h1 >= min_count) and (h2 >= min_count) and (both >= min_count)
    if mode == "AD":
        h2 = int(np.sum(g2 == 1))
        if h2 < min_count:
            return False
        for gi in (0, 1, 2):
            m = (g1 == gi)
            if int(m.sum()) == 0:
                continue
            if int(np.sum(m & (g2 == 1))) < min_count // 2:
                return False
        return True
    # MULT / XOR / AA / threshold: 3x3 joint cells. Empty cells are allowed,
    # but a non-empty cell must not be too small
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            c = int(np.sum((g1 == a) & (g2 == b)))
            if 0 < c < min_count // 2:
                return False
    return True

def interaction_F_conditioned(y, inter, mains):
    """
    Null model = intercept + all main effects (mains); full model = null + inter.
    Returns the F statistic of the interaction term.
    mains already contains the structure PCs, the large main-effect loci and
    the two local main effects, so this is the interaction after structure and
    main effects have been controlled for.
    """
    n = len(y)
    Xr = np.column_stack([np.ones(n)] + mains)
    Xf = np.column_stack([np.ones(n)] + mains + [inter])
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    rss_r = float(np.sum((y - Xr @ br) ** 2))
    bf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    rss_f = float(np.sum((y - Xf @ bf) ** 2))
    df2 = n - Xf.shape[1]
    if rss_f <= 1e-12 or df2 <= 0:
        return 0.0
    return max(((rss_r - rss_f) / 1.0) / (rss_f / df2), 0.0)

def interaction_effect_conditioned(y, inter, mains):
    """As above, additionally returning the interaction coefficient and its SE."""
    n = len(y)
    Xf = np.column_stack([np.ones(n)] + mains + [inter])
    bf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    resid = y - Xf @ bf
    df2 = n - Xf.shape[1]
    if df2 <= 0:
        return bf[-1], np.nan
    s2 = float(np.sum(resid ** 2)) / df2
    try:
        se = np.sqrt(max(s2 * np.linalg.inv(Xf.T @ Xf)[-1, -1], 0.0))
    except np.linalg.LinAlgError:
        se = np.nan
    return bf[-1], se


# =====================================================================
#  The AA (additive x additive) and threshold tests, built on the same
#  interaction_F_conditioned framework as AD / DD / MULT / XOR: NOIA coding,
#  main-effect conditioning and the genotype-cell filter.
# =====================================================================
def _minor_dose(Xa_col, af1):
    """Convert an A1 dose (0/1/2) to a minor-allele dose, so that count and
    carrier definitions do not depend on which allele was coded A1."""
    return Xa_col if af1 < 0.5 else (2 - Xa_col)


def score_pair(i, j, y, Xa, Ha, Hd, af1, global_mains, types_to_compute=None,
               thr_ks=CFG.THRESHOLD_K_RANGE, min_count=MIN_CELL_COUNT):
    """
    Score one pair (i, j) under all six interaction tests: F statistic,
    interaction coefficient and SE, and a flag for whether the genotype cells
    support the test.

    global_mains  structure PCs and large main-effect loci, one vector per
                  column, entering every null model
    Each of the six interaction terms is added as one further column on top of
    [global_mains + Ha_i, Hd_i, Ha_j, Hd_j] and tested by F. Returns a dict
    ready for a DataFrame. No threshold or significance decision is made here.

    """
    g1 = Xa[:, i]; g2 = Xa[:, j]
    hai, hdi = Ha[:, i], Hd[:, i]
    haj, hdj = Ha[:, j], Hd[:, j]
    loc_mains = [hai, hdi, haj, hdj]
    mains = global_mains + loc_mains

    T = None if types_to_compute is None else set(types_to_compute)
    def _want(t):
        return (T is None) or (t in T)

    out = {"snp_i": i, "snp_j": j}

    def _one(inter, mode, tag):
        ok = _cells_ok(g1, g2, mode, min_count)
        if not ok:
            out[f"F_{tag}"] = 0.0
            out[f"beta_{tag}"] = np.nan
            out[f"se_{tag}"] = np.nan
            out[f"testable_{tag}"] = False
            return
        F = interaction_F_conditioned(y, inter, mains)
        b, se = interaction_effect_conditioned(y, inter, mains)
        out[f"F_{tag}"] = float(F)
        out[f"beta_{tag}"] = float(b)
        out[f"se_{tag}"] = float(se)
        out[f"testable_{tag}"] = True

    # AA, additive x additive (NOIA): Ha_i * Ha_j
    if _want("AA"):
        _one(hai * haj, "MULT", "AA")
    # AD, additive x dominance (NOIA); the orientation convention is ha_i * Hd_j
    if _want("AD"):
        _one(hai * hdj, "AD", "AD")
    # DD, dominance x dominance (NOIA): Hd_i * Hd_j
    if _want("DD"):
        _one(hdi * hdj, "DD", "DD")
    # MULT, multiplicative: the centred product of the additive doses
    if _want("MULT"):
        xm = (g1 - g1.mean()) * (g2 - g2.mean())
        _one(xm, "MULT", "MULT")
    # Threshold: the best count>=k contrast, scanning k.
    #   The null model here differs from AA/AD/DD/MULT on purpose. It conditions on
    #   structure, the large main-effect loci and the additive main effects Ha of the
    #   two loci, but NOT on the dominance main effects Hd: Hd would absorb the
    #   count>=k step itself and drive a real threshold interaction to zero. What is
    #   tested is whether crossing threshold k still shifts the group mean once
    #   the additive dose has been controlled for: the non-linearity specific
    #   to a threshold model
    md1 = _minor_dose(g1, af1[i]); md2 = _minor_dose(g2, af1[j])
    if _want("THRESH"):
      thr_mains = global_mains + [hai, haj]
      cnt = md1 + md2
      bestF, bestk, bestb, bestse, testable = 0.0, np.nan, np.nan, np.nan, _cells_ok(g1, g2, "MULT", min_count)
      if testable:
        for k in thr_ks:
            ind = (cnt >= k).astype(float)
            if ind.std() < 1e-9:      # no contrast at this k (all 0 or all 1)
                continue
            F = interaction_F_conditioned(y, ind - ind.mean(), thr_mains)
            if F > bestF:
                b, se = interaction_effect_conditioned(y, ind - ind.mean(), thr_mains)
                bestF, bestk, bestb, bestse = float(F), int(k), float(b), float(se)
      out["F_THRESH"] = bestF
      out["beta_THRESH"] = bestb
      out["se_THRESH"] = bestse
      out["best_k_THRESH"] = bestk
      out["testable_THRESH"] = bool(testable)
    # XOR: carrier_i XOR carrier_j (exactly one locus carries the minor
    # allele). No marginal effect is required.
    if _want("XOR"):
        ci = (md1 >= 1).astype(int); cj = (md2 >= 1).astype(int)
        xor = ((ci + cj) == 1).astype(float)
        if xor.std() < 1e-9:
            out["F_XOR"] = 0.0; out["beta_XOR"] = np.nan; out["se_XOR"] = np.nan
            out["testable_XOR"] = False
        else:
            _one(xor - xor.mean(), "MULT", "XOR")
    return out


# =====================================================================
#  Pure-Python PLINK .bed reader (SNP-major, 2-bit); avoids a pandas_plink dependency
# =====================================================================
def normalize_id(s):
    s = str(s).strip().lstrip("\ufeff").replace("\r", "").replace("\n", "")
    if s.endswith(".0") and s[:-2].replace("-", "").replace("_", "").isalnum():
        s = s[:-2]
    return s

def read_bim(prefix):
    df = pd.read_csv(prefix + ".bim", sep=r"\s+", header=None,
                     names=["chrom", "snp", "cm", "pos", "a1", "a2"], dtype=str)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce").fillna(0).astype(np.int64)
    return df

def read_fam(prefix):
    df = pd.read_csv(prefix + ".fam", sep=r"\s+", header=None,
                     names=["fid", "iid", "pat", "mat", "sex", "phe"], dtype=str)
    # IIDs are never rewritten or normalised: phenotypes and covariates are
    # matched on the raw identifiers as they appear in the .fam file
    return df

def read_bed(prefix, n_ind, n_snp):
    """SNP-major .bed -> (n_ind, n_snp) additive dose 0/1/2; missing = -1."""
    with open(prefix + ".bed", "rb") as f:
        magic = f.read(3)
        if magic[:2] != b"\x6c\x1b":
            raise ValueError("not a valid PLINK .bed file")
        if magic[2] != 1:
            raise ValueError("only SNP-major .bed (mode=1) is supported")
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    bytes_per_snp = (n_ind + 3) // 4
    raw = raw.reshape(n_snp, bytes_per_snp)
    # 2-bit decode: PLINK uses 00 = hom A1 (2), 01 = missing, 10 = het (1), 11 = hom A2 (0)
    lut = np.array([2, -1, 1, 0], dtype=np.int8)
    X = np.empty((n_snp, n_ind), dtype=np.int8)
    for b in range(4):
        vals = lut[(raw >> (2 * b)) & 0b11]     # (n_snp, bytes_per_snp)
        idx = np.arange(b, n_ind, 4)
        if len(idx) == 0:
            continue
        X[:, idx] = vals[:, :len(idx)]
    return X.T.astype(float)                     # (n_ind, n_snp)


# =====================================================================
#  Part 0a  ADMIXTURE structure (optional; falls back to PCA+KMeans if absent)
# =====================================================================
def run_admixture_cv(bfile, k_range, workdir, plink_bin, admix_bin, threads=8):
    if shutil.which(plink_bin) is None or shutil.which(admix_bin) is None:
        return None, None
    os.makedirs(workdir, exist_ok=True)
    bed = os.path.join(workdir, "ref")
    try:
        subprocess.run([plink_bin, "--bfile", bfile, "--make-bed", "--allow-extra-chr",
                        "--out", bed], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # ADMIXTURE only accepts integer chromosome codes
        bim = bed + ".bim"; lines = open(bim).read().splitlines()
        cmap = {}; outl = []
        for ln in lines:
            fld = ln.split("\t") if "\t" in ln else ln.split()
            if not fld[0].lstrip("-").isdigit():
                cmap.setdefault(fld[0], len(cmap) + 1); fld[0] = str(cmap[fld[0]])
            outl.append("\t".join(fld))
        open(bim, "w").write("\n".join(outl) + "\n")
        cv = {}
        for k in k_range:
            logf = os.path.join(workdir, f"k{k}.log")
            qf = os.path.join(workdir, f"ref.{k}.Q")
            # reuse: if this K already finished (a .Q file exists and the log
            # contains a CV error), read it instead of rerunning
            if os.path.exists(qf) and os.path.exists(logf):
                got = None
                for line in open(logf, errors="ignore"):
                    mm = re.search(r"CV error \(K=(\d+)\):\s*([\d.]+)", line)
                    if mm:
                        got = float(mm.group(2))
                if got is not None:
                    cv[k] = got
                    print(f"      [admix] K={k} already done, skipping (CV={got:.5f})")
                    continue
            print(f"      [admix] running K={k} (-j{threads}) ...", flush=True)
            try:
                with open(logf, "w") as fo:
                    subprocess.run([admix_bin, "--cv", f"-j{threads}", "-s", "1",
                                    "ref.bed", str(k)],
                                   cwd=workdir, check=True, stdout=fo,
                                   stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                # a failure at one K skips that K only; other K values stand
                print(f"      [admix] K={k} failed, skipped (other K unaffected)")
                continue
            for line in open(logf, errors="ignore"):
                mm = re.search(r"CV error \(K=(\d+)\):\s*([\d.]+)", line)
                if mm:
                    cv[int(mm.group(1))] = float(mm.group(2))
        if not cv:
            return None, None
        return cv, _pick_admix_k(cv, workdir)
    except subprocess.CalledProcessError:
        return None, None


def _pick_admix_k(cv, workdir, flat_total_drop=0.05):
    """
    The K with the smallest CV error cannot be taken directly. In a closed herd
    the CV error often falls monotonically by a tiny amount, which is itself
    evidence that the population has no discrete strata; min() then always
    selects the largest K tested, manufacturing clusters that cannot be
    defended. Decision rule: if the curve falls monotonically over the tested
    range with no interior minimum, there is no discrete structure and K=1.
    """
    ks = sorted(cv)
    if len(ks) == 1:
        k = ks[0]
    else:
        vals = [cv[k] for k in ks]
        imin = int(np.argmin(vals))
        # an interior minimum means a genuine elbow: take it
        if 0 < imin < len(ks) - 1:
            k = ks[imin]
        elif imin == len(ks) - 1:
            # the minimum sits at the largest K: the curve is still falling, no elbow
            rel = (vals[0] - vals[-1]) / max(abs(vals[0]), 1e-12)
            if rel < flat_total_drop:
                print(f"      [admix] CV error falls monotonically with no "
                      f"elbow and a total drop of only {rel*100:.1f}% "
                      f"(<{flat_total_drop*100:.0f}%) => no discrete structure; "
                      f"K=1 (a stratified split equals a random one)")
                k = ks[0]
            else:
                print(f"      [admix] CV error is still smallest at the "
                      f"largest K ({ks[-1]}) with no elbow, but the total drop "
                      f"reaches {rel*100:.1f}%: the optimum may lie beyond the "
                      f"tested range. Taking K={ks[-1]}; widen --admix_kmax to "
                      f"check.")
                k = ks[-1]
        else:
            k = ks[0]
    qf = os.path.join(workdir, f"ref.{k}.Q")
    Q = np.loadtxt(qf) if (k > 1 and os.path.exists(qf)) else None
    return (k, Q)


def structure_clusters(K_A, admix_out, fallback_max_k, seed):
    """Cluster label per individual: ADMIXTURE argmax Q where available, else PCA+KMeans."""
    if admix_out is not None and admix_out[0] == 1:
        return np.zeros(K_A.shape[0], dtype=int), ("admixture_no_structure", 1)
    if admix_out is not None and admix_out[1] is not None:
        Q = admix_out[1]
        if Q.ndim == 1:
            Q = Q[:, None]
        return np.argmax(Q, axis=1), ("admixture", admix_out[0])
    # fallback: PCA + KMeans, with K chosen by silhouette over [1..fallback_max_k]
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    PC = pca_from_grm(K_A, min(10, K_A.shape[0] - 1))
    if PC.shape[1] == 0:
        return np.zeros(K_A.shape[0], dtype=int), ("single", 1)
    best_k, best_s, best_lab = 1, -1, np.zeros(K_A.shape[0], dtype=int)
    for k in range(2, fallback_max_k + 1):
        if k >= K_A.shape[0]:
            break
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(PC[:, :5])
        try:
            s = silhouette_score(PC[:, :5], lab)
        except Exception:
            s = -1
        if s > best_s:
            best_k, best_s, best_lab = k, s, lab
    # a low silhouette means no real discrete structure: one cluster, which
    # makes the stratified split equivalent to a random one
    if best_s < 0.25:
        return np.zeros(K_A.shape[0], dtype=int), ("pca_kmeans_flat", 1)
    return best_lab, ("pca_kmeans", int(best_k))


def stratified_split(labels, test_frac, seed):
    """Cluster-stratified train/test split preserving cluster shares. Returns is_test."""
    rng = np.random.RandomState(seed)
    n = len(labels); is_test = np.zeros(n, dtype=bool)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(len(idx) * test_frac))) if len(idx) > 4 else 0
        is_test[idx[:n_te]] = True
    return is_test


def stratified_kfold(labels, n_folds, seed):
    """Cluster-stratified K-fold indices: a length-n vector of fold numbers 0..K-1."""
    rng = np.random.RandomState(seed + 7)
    n = len(labels); fold = np.full(n, -1, dtype=int)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]; rng.shuffle(idx)
        for pos, ii in enumerate(idx):
            fold[ii] = pos % n_folds
    # if a cluster is so small that a fold ends up empty, reshuffle as a fallback
    if np.any(np.bincount(fold, minlength=n_folds) == 0):
        order = rng.permutation(n)
        fold = np.array([order.tolist().index(i) % n_folds for i in range(n)])
    return fold


# =====================================================================
#  Part 1  Genomic prediction (multi-kernel GBLUP REML): K-fold CV + held-out test
# =====================================================================
# The prediction models, i.e. which kernels each one uses.
PRED_MODELS = {
    "A":            ["A"],
    "A+D":          ["A", "D"],
    "A+D+AA":       ["A", "D", "AA"],           # the model the epistatic gain is read from
    "A+D+AA+DD+AD": ["A", "D", "AA", "DD", "AD"],
}
EPI_MODEL   = "A+D+AA"     # epistatic gain = this model minus the baseline
BASE_MODEL  = "A+D"        # the correct baseline for "is epistasis useful" is A+D, not A


def _sub(K, r, c):
    return K[np.ix_(r, c)]


def predict_fold(y, Xfix, Kdict, tr_idx, te_idx, model_keys):
    """Fit REMLgrad on tr, predict te; return predictions and variance fractions."""
    Ks_tt = [_sub(Kdict[k], tr_idx, tr_idx) for k in model_keys]
    f = REMLgrad().fit(y[tr_idx], Xfix[tr_idx], Ks_tt)
    Ks_new_t = [_sub(Kdict[k], te_idx, tr_idx) for k in model_keys]
    pred = f.predict_new(y[tr_idx], Xfix[tr_idx], Ks_tt, Xfix[te_idx], Ks_new_t)
    v = f.variances_; prop = v / v.sum()
    return pred, prop, f.loglik_


def run_cv_prediction(y, Xfix, Kdict, labels_tr, cfg):
    """K-fold CV inside the training set: per-fold PCC per model, mean +/- SD, and dPCC."""
    fold = stratified_kfold(labels_tr, cfg.N_CV_FOLDS, cfg.SEED)
    ntr = len(y)
    rows = []; delta_rows = []
    prop_acc = {m: [] for m in PRED_MODELS}
    for kf in range(cfg.N_CV_FOLDS):
        te_idx = np.where(fold == kf)[0]
        tr_idx = np.where(fold != kf)[0]
        preds = {}
        for m, keys in PRED_MODELS.items():
            pred, prop, _ = predict_fold(y, Xfix, Kdict, tr_idx, te_idx, keys)
            preds[m] = pred
            prop_acc[m].append(prop)
            pcc = pearsonr(pred, y[te_idx])[0]
            rows.append({"fold": kf, "model": m, "test_PCC": pcc, "n_test": len(te_idx)})
        # epistatic gain, fold by fold
        d_epi = pearsonr(preds[EPI_MODEL], y[te_idx])[0] - pearsonr(preds[BASE_MODEL], y[te_idx])[0]
        d_full = pearsonr(preds["A+D+AA+DD+AD"], y[te_idx])[0] - pearsonr(preds[BASE_MODEL], y[te_idx])[0]
        d_dom = pearsonr(preds["A+D"], y[te_idx])[0] - pearsonr(preds["A"], y[te_idx])[0]
        delta_rows.append({"fold": kf,
                           "dPCC_dominance(A+D - A)": d_dom,
                           "dPCC_epistasis(A+D+AA - A+D)": d_epi,
                           "dPCC_full_epi(+DD+AD - A+D)": d_full})
    cv_df = pd.DataFrame(rows)
    summ = (cv_df.groupby("model")["test_PCC"]
            .agg(["mean", "std"]).reindex(list(PRED_MODELS)).reset_index()
            .rename(columns={"mean": "PCC_mean", "std": "PCC_sd"}))
    delta_df = pd.DataFrame(delta_rows)
    delta_summ = delta_df.drop(columns="fold").agg(["mean", "std"]).T.reset_index()
    delta_summ.columns = ["contrast", "dPCC_mean", "dPCC_sd"]
    return cv_df, summ, delta_df, delta_summ, prop_acc


def run_holdout_prediction(y, Xfix, Kdict, tr_idx, te_idx, cfg):
    """Fit on the whole training set, predict the held-out test set; dPCC with a paired-bootstrap 95% CI."""
    preds = {}; props = {}; logls = {}
    for m, keys in PRED_MODELS.items():
        pred, prop, ll = predict_fold(y, Xfix, Kdict, tr_idx, te_idx, keys)
        preds[m] = pred; props[m] = prop; logls[m] = ll
    yte = y[te_idx]
    rows = []
    for m in PRED_MODELS:
        rows.append({"model": m, "holdout_PCC": pearsonr(preds[m], yte)[0]})
    # paired bootstrap
    rs = np.random.RandomState(cfg.SEED + 99)
    idxm = rs.randint(0, len(yte), (cfg.N_BOOT, len(yte)))
    def bp(p):
        return np.array([pearsonr(p[ix], yte[ix])[0] for ix in idxm])
    base = bp(preds[BASE_MODEL])
    delta_rows = []
    for m in [EPI_MODEL, "A+D+AA+DD+AD"]:
        d = bp(preds[m]) - base; d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        pt = pearsonr(preds[m], yte)[0] - pearsonr(preds[BASE_MODEL], yte)[0]
        delta_rows.append({"contrast": f"{m} - {BASE_MODEL}",
                           "dPCC": pt, "CI_low": lo, "CI_high": hi,
                           "excludes_0": bool(not (lo <= 0 <= hi))})
    return pd.DataFrame(rows), pd.DataFrame(delta_rows), props, logls


# =====================================================================
#  Evaluation metrics (the set below applies to continuous traits)
# =====================================================================
def eval_metrics(pred, obs, top_frac=0.10, overlap_frac=0.20):
    """
    A balanced evaluation for a continuous trait:
      PCC      the common currency; insensitive to linear rescaling
      R2       sensitive to scale and offset, so it exposes calibration problems
      Spearman breeding decisions are about rank, not value
      top_gain mean phenotype of the selected top_frac minus the overall mean,
               i.e. the realised gain from keeping those animals
      overlap  overlap between the predicted and the observed top fraction, i.e.
               precision at a fixed selection intensity. Preferred to F1, which
               would require dichotomising a continuous trait arbitrarily
      slope    calibration slope from regressing observed on predicted; 1 is ideal
    """
    pred = np.asarray(pred, float); obs = np.asarray(obs, float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[ok], obs[ok]
    n = len(obs)
    out = {}
    out["PCC"] = float(pearsonr(pred, obs)[0]) if n > 2 and pred.std() > 0 else np.nan
    ss_res = float(((obs - pred) ** 2).sum()); ss_tot = float(((obs - obs.mean()) ** 2).sum())
    out["R2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    out["Spearman"] = float(spearmanr(pred, obs)[0]) if n > 2 else np.nan
    k = max(1, int(round(n * top_frac)))
    sel = np.argsort(-pred)[:k]
    out["top_gain"] = float(obs[sel].mean() - obs.mean())
    ko = max(1, int(round(n * overlap_frac)))
    out["top_overlap"] = float(len(np.intersect1d(np.argsort(-pred)[:ko],
                                                  np.argsort(-obs)[:ko])) / ko)
    if pred.std() > 1e-12:
        out["calib_slope"] = float(np.polyfit(pred, obs, 1)[0])
    else:
        out["calib_slope"] = np.nan
    return out


def bh_fdr(pvals):
    """Benjamini-Hochberg correction; returns q values."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    pp = p[ok]; n = len(pp)
    if n == 0:
        return q
    order = np.argsort(pp)
    ranked = pp[order]
    qq = ranked * n / (np.arange(n) + 1.0)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    qq = np.clip(qq, 0, 1)
    out = np.empty(n); out[order] = qq
    q[ok] = out
    return q


# =====================================================================
#  Multi-resolution slice kernels: construction
#    - The trans (cross-chromosome) AA / DD / AD kernels are computed EXACTLY by
#      chromosome decomposition, with no pair enumeration:
#        sum over all (i,j)      = (M M^T) o (N N^T)
#        sum within chromosome c = (M_c M_c^T) o (N_c N_c^T)
#        cross-chromosome        = all - sum over c
#      cis kernels are enumerated exactly inside the window.
#    - Threshold and XOR admit no such factorisation, so they use a fixed-seed
#      Monte-Carlo sample of pairs, labelled as such in the output.
#    Every slice kernel is built from X / chrom / pos / af1 alone and never
#    from the phenotype, so leakage is structurally impossible.
# =====================================================================
def _sym_norm(K):
    d = np.mean(np.diag(K))
    K = K / d if d > 1e-12 else K
    # cis slice kernels are often rank-deficient (fewer pairs than individuals),
    # so a tiny jitter keeps REML numerically stable
    return K + np.eye(K.shape[0]) * 1e-6


class GramCache:
    """
    The AA kernel needs Ha@Ha.T, DD needs Hd@Hd.T and AD needs both, each once
    genome-wide and once per chromosome. At n=2796, m=35750 one such product is
    about 2.8e11 flops, so Gram matrices are cached per (matrix, scope) and
    """
    def __init__(self, chrom):
        self.chrom = chrom
        self._full = {}
        self._perchr = {}

    def full(self, key, M):
        if key not in self._full:
            self._full[key] = M @ M.T
        return self._full[key]

    def per_chrom(self, key, M):
        if key not in self._perchr:
            d = {}
            for c in np.unique(self.chrom):
                sidx = np.where(self.chrom == c)[0]
                if len(sidx):
                    d[c] = M[:, sidx] @ M[:, sidx].T
            self._perchr[key] = d
        return self._perchr[key]

    def free(self):
        self._full.clear(); self._perchr.clear()


def kern_prod_exact(M, N, chrom, mode, gram=None, key_m="M", key_n="N"):
    """
    Between-individual covariance kernel of sum_{i,j} (M_i o N_j).
    mode is 'trans' (cross-chromosome) or 'within' (same chromosome). Uses the
    (MM^T) o (NN^T) identity, so it is exact and never enumerates pairs; the
    Gram matrices come from GramCache.
    """
    if gram is None:
        gram = GramCache(chrom)
    tot = gram.full(key_m, M) * gram.full(key_n, N)
    gm = gram.per_chrom(key_m, M); gn = gram.per_chrom(key_n, N)
    within = np.zeros_like(tot)
    for c in gm:
        if c in gn:
            within += gm[c] * gn[c]
    K = (tot - within) if mode == "trans" else within
    return _sym_norm(K)


def kern_from_pairs(pairs, feat_fn_vec, n, chunk=20000, verbose=False):
    """
    Build a kernel from an explicit pair list: K = W W^T / n_pairs.
    feat_fn_vec processes a batch of pairs at a time rather than one pair per
    call: cis pair counts reach the millions and a Python loop would dominate.
    """
    P = np.asarray(pairs, dtype=np.int64)
    if len(P) == 0:
        return None
    K = np.zeros((n, n))
    cnt = 0
    nb = (len(P) + chunk - 1) // chunk
    for b0 in range(0, len(P), chunk):
        blk = P[b0:b0 + chunk]
        W = feat_fn_vec(blk[:, 0], blk[:, 1])          # (n, n_pairs_in_block)
        W = W - W.mean(0, keepdims=True)
        sd = W.std(0)
        good = sd > 1e-12
        if not good.any():
            continue
        W = W[:, good] / sd[good]
        K += W @ W.T
        cnt += int(good.sum())
        if verbose and nb > 1:
            print(f"      ...{min(b0+chunk, len(P))}/{len(P)} pairs", flush=True)
    if cnt == 0:
        return None
    return _sym_norm(K / cnt)


def cis_pair_list(chrom, pos, window_bp, max_neighbors, snp_subset=None):
    """All same-chromosome pairs within `window`, keeping at most max_neighbors per anchor."""
    idx_all = np.arange(len(chrom)) if snp_subset is None else np.asarray(snp_subset)
    pairs = set()
    for c in np.unique(chrom[idx_all]):
        s = idx_all[chrom[idx_all] == c]
        s = s[np.argsort(pos[s])]
        p = pos[s]
        for a in range(len(s)):
            hi = np.searchsorted(p, p[a] + window_bp, side="right")
            nb = s[a + 1:hi]
            if len(nb) > max_neighbors:
                nb = nb[:max_neighbors]
            for b in nb:
                pairs.add((int(min(s[a], b)), int(max(s[a], b))))
    return sorted(pairs)


def trans_pair_sample(chrom, n_pairs, rng, snp_subset=None):
    """Random cross-chromosome pair sample, fixed seed."""
    idx_all = np.arange(len(chrom)) if snp_subset is None else np.asarray(snp_subset)
    if len(idx_all) < 2:
        return []
    out = set()
    tries = 0
    while len(out) < n_pairs and tries < n_pairs * 20:
        i = idx_all[rng.randint(len(idx_all))]
        j = idx_all[rng.randint(len(idx_all))]
        tries += 1
        if i == j or chrom[i] == chrom[j]:
            continue
        out.add((int(min(i, j)), int(max(i, j))))
    return sorted(out)


def xor_eligible(X, af1, cfg):
    """Pre-filter for the XOR pool: MAF >= XOR_MIN_MAF and heterozygosity >= XOR_MIN_HET."""
    maf = np.minimum(af1, 1 - af1)
    het = (X == 1).mean(0)
    return np.where((maf >= cfg.XOR_MIN_MAF) & (het >= cfg.XOR_MIN_HET))[0]


def build_slice_kernels(X, Ha, Hd, chrom, pos, af1, cfg, verbose=True):
    """
    Build the panel of ten slice kernels: {cis, trans} x {AA, DD, AD, threshold, XOR}.
    Returns a dict of name -> (kernel, description of how it was built).
    """
    n = X.shape[0]
    rng = np.random.RandomState(cfg.SEED)
    win = int(cfg.CIS_WINDOW_MB * 1e6)
    out = {}

    # ---- AA / DD / AD: exact ----
    gram = GramCache(chrom)
    for name, (M, N, km, kn) in {"AA": (Ha, Ha, "Ha", "Ha"),
                                 "DD": (Hd, Hd, "Hd", "Hd"),
                                 "AD": (Ha, Hd, "Ha", "Hd")}.items():
        if verbose:
            print(f"    building trans_{name} (chromosome decomposition, cached Gram)...", flush=True)
        out[f"trans_{name}"] = (kern_prod_exact(M, N, chrom, "trans", gram, km, kn),
                                "exact (chromosome decomposition)")
    gram.free()
    cis_pairs = cis_pair_list(chrom, pos, win, cfg.MAX_CIS_NEIGHBORS)
    if verbose:
        print(f"  pairs within the {cfg.CIS_WINDOW_MB}Mb cis window: {len(cis_pairs)}")
    for name, fn in {
        "AA": lambda I, J: Ha[:, I] * Ha[:, J],
        "DD": lambda I, J: Hd[:, I] * Hd[:, J],
        "AD": lambda I, J: Ha[:, I] * Hd[:, J],
    }.items():
        if verbose:
            print(f"    building cis_{name} ({len(cis_pairs)} pairs)...", flush=True)
        K = kern_from_pairs(cis_pairs, fn, n, verbose=verbose)
        if K is not None:
            out[f"cis_{name}"] = (K, f"exact ({len(cis_pairs)} pairs enumerated in window)")

    # ---- threshold / XOR: no factorisation exists, so a fixed-seed sample is
    # used and labelled honestly as a Monte-Carlo approximation ----
    md = np.where(af1[None, :] < 0.5, X, 2 - X)          # minor-allele dose
    carrier = (md >= 1).astype(float)

    def f_thr(I, J):
        return ((md[:, I] + md[:, J]) >= cfg.KERNEL_THRESH_K).astype(float)

    def f_xor(I, J):
        return ((carrier[:, I] + carrier[:, J]) == 1).astype(float)

    xpool = xor_eligible(X, af1, cfg)
    cis_thr = cis_pairs
    cis_xor = cis_pair_list(chrom, pos, win, cfg.MAX_CIS_NEIGHBORS, snp_subset=xpool)
    tr_thr = trans_pair_sample(chrom, cfg.KERNEL_TRANS_SAMPLE, rng)
    tr_xor = trans_pair_sample(chrom, cfg.KERNEL_TRANS_SAMPLE, rng, snp_subset=xpool)
    for nm, prs, fn in [("cis_THRESH", cis_thr, f_thr), ("trans_THRESH", tr_thr, f_thr),
                        ("cis_XOR", cis_xor, f_xor), ("trans_XOR", tr_xor, f_xor)]:
        if verbose:
            print(f"    building {nm} ({len(prs)} pairs)...", flush=True)
        K = kern_from_pairs(prs, fn, n, verbose=verbose)
        if K is not None:
            tag = ("exact (enumerated in window)" if nm.startswith("cis")
                   else f"Monte-Carlo approximation ({len(prs)} pairs sampled)")
            out[nm] = (K, tag)
    return out


# =====================================================================
#  The gate: an active attempt to falsify the diffuse hypothesis
#    The slice kernels deliberately look for concentrated signal at several
#    resolutions. If none of them passes, that is positive evidence FOR a
#    diffuse architecture, not merely a way of discarding weak candidates.
#    The final conclusion still comes only from the held-out test set.
#    Rule: paired-bootstrap CI on dPCC, BH-FDR (q < Q), and cross-fold
#    stability, all three. The classification below is fixed in advance so
#    that no result can be selected after the fact:
#      significant and above the global aggregate kernel -> concentrated
#      significant but not above it                      -> diffuse-compatible
#      not significant                                   -> negative
# =====================================================================
def _pooled_oof(y, Xfix, Kdict, labels, cfg, model_keys):
    """Out-of-fold predictions inside the training set (each individual once) plus per-fold PCC."""
    fold = stratified_kfold(labels, cfg.N_CV_FOLDS, cfg.SEED)
    oof = np.full(len(y), np.nan)
    per_fold = []
    for kf in range(cfg.N_CV_FOLDS):
        te = np.where(fold == kf)[0]; tr = np.where(fold != kf)[0]
        if len(te) < 3 or len(tr) < 10:
            continue
        try:
            pred, _, _ = predict_fold(y, Xfix, Kdict, tr, te, model_keys)
            oof[te] = pred
            per_fold.append(pearsonr(pred, y[te])[0] if np.std(pred) > 0 else np.nan)
        except Exception:
            per_fold.append(np.nan)
    return oof, np.array(per_fold, float)


class _GateCfg:
    """A reduced REML budget for the gate. The gate only screens and never
    concludes; the frozen test set remains the judge, so iterations and
    restarts are capped here to control cost. Part 1 and the final validation
    use the full budget.
    """
    def __init__(self, cfg):
        for k in dir(cfg):
            if not k.startswith("__"):
                setattr(self, k, getattr(cfg, k))
        self.REML_MAXITER = min(getattr(cfg, "REML_MAXITER", 100), cfg.GATE_REML_MAXITER)
        self.REML_RESTARTS = 1


def _gate_one(name, K, how, y_tr, Xfix_tr, Kbase_tr, labels_tr, cfg,
              base_oof, base_folds, seed):
    """
    Gate computation for one slice kernel; safe to run in parallel.

    The bootstrap seed is offset by the kernel index. Without that offset every
    worker would draw the same resamples and every kernel would report an
    identical confidence interval -- a mistake that hides very well.
    """
    rng = np.random.RandomState(seed)
    Kd = dict(Kbase_tr); Kd["S"] = K
    try:
        s_oof, s_folds = _pooled_oof(y_tr, Xfix_tr, Kd, labels_tr, cfg, ["A", "D", "S"])
    except Exception as e:
        return {"slice": name, "build": how, "dPCC": np.nan, "CI_low": np.nan,
                "CI_high": np.nan, "p_boot": np.nan, "fold_sd": np.nan,
                "stability_cv": np.nan, "note": f"fit failed: {e}"}
    ok = np.isfinite(base_oof) & np.isfinite(s_oof)
    if ok.sum() < 10:
        return {"slice": name, "build": how, "dPCC": np.nan, "CI_low": np.nan,
                "CI_high": np.nan, "p_boot": np.nan, "fold_sd": np.nan,
                "stability_cv": np.nan, "note": "too few valid out-of-fold predictions"}
    d_point = pearsonr(s_oof[ok], y_tr[ok])[0] - pearsonr(base_oof[ok], y_tr[ok])[0]
    idx = np.where(ok)[0]
    boots = np.empty(cfg.N_BOOT)
    for b in range(cfg.N_BOOT):
        sidx = idx[rng.randint(0, len(idx), len(idx))]
        if np.std(y_tr[sidx]) < 1e-12:
            boots[b] = np.nan; continue
        boots[b] = (pearsonr(s_oof[sidx], y_tr[sidx])[0]
                    - pearsonr(base_oof[sidx], y_tr[sidx])[0])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan)
    p_boot = 2 * min((boots <= 0).mean(), (boots >= 0).mean()) if len(boots) else np.nan
    p_boot = min(1.0, max(p_boot, 1.0 / max(len(boots), 1)))
    df_folds = s_folds - base_folds if len(s_folds) == len(base_folds) else np.array([np.nan])
    df_folds = df_folds[np.isfinite(df_folds)]
    fold_sd = float(np.std(df_folds)) if len(df_folds) > 1 else np.nan
    stab_cv = (float(fold_sd / abs(np.mean(df_folds)))
               if len(df_folds) > 1 and abs(np.mean(df_folds)) > 1e-9 else np.inf)
    return {"slice": name, "build": how, "dPCC": d_point, "CI_low": lo, "CI_high": hi,
            "p_boot": p_boot, "fold_sd": fold_sd, "stability_cv": stab_cv, "note": ""}


def run_gating(y_tr, Xfix_tr, Kbase_tr, slice_K_tr, labels_tr, cfg, global_dpcc,
               ckpt_cfg=None, fp=None):
    """The gate is the most expensive stage, so each slice kernel is cached
    separately: a crash on the seventh kernel does not cost the first six."""
    cfg = _GateCfg(cfg)
    """
    For each slice kernel: out-of-fold CV dPCC of A+D+slice against A+D, a
    paired-bootstrap CI, and cross-fold stability. Returns the gate DataFrame.
    """
    rng = np.random.RandomState(cfg.SEED)
    ntr = len(y_tr)
    cached = ck_load(ckpt_cfg, "gate_base_oof", fp) if ckpt_cfg is not None else None
    if cached is None:
        base_oof, base_folds = _pooled_oof(y_tr, Xfix_tr, Kbase_tr, labels_tr, cfg, ["A", "D"])
        if ckpt_cfg is not None:
            ck_save(ckpt_cfg, "gate_base_oof", fp, (base_oof, base_folds))
    else:
        base_oof, base_folds = cached
    # ---- kernels not already cached ----
    todo = []
    rows = []
    for name, (K, how) in slice_K_tr.items():
        done = ck_load(ckpt_cfg, f"gate_slice_{name}", fp) if ckpt_cfg is not None else None
        if done is not None:
            rows.append(done)
        else:
            todo.append((name, K, how))

    # ---- the ten slice kernels are independent, so run them in parallel,
    # limiting BLAS threads per worker so they do not contend ----
    if _HAS_JOBLIB and getattr(cfg, "GATE_N_JOBS", 1) != 1 and len(todo) > 1:
        nj = min(len(todo), cpu_count() if cfg.GATE_N_JOBS in (-1, None)
                 else int(cfg.GATE_N_JOBS))
        print(f"    gate in parallel: {len(todo)} kernel(s) to compute, "
              f"{nj} process(es), {max(1, cpu_count()//max(nj,1))} BLAS "
              f"thread(s) each", flush=True)
        inner = max(1, cpu_count() // max(nj, 1))
        with par_ctx(inner):
            out = Parallel(n_jobs=nj, backend="loky")(
                delayed(_gate_one)(name, K, how, y_tr, Xfix_tr, Kbase_tr, labels_tr,
                                   cfg, base_oof, base_folds, cfg.SEED + gi)
                for gi, (name, K, how) in enumerate(todo))
        for row in out:
            rows.append(row)
            if ckpt_cfg is not None:
                ck_save(ckpt_cfg, f"gate_slice_{row['slice']}", fp, row)
    else:
        for gi, (name, K, how) in enumerate(todo):
            print(f"    [gate {gi+1}/{len(todo)}] {name} ...", flush=True)
            _t0 = time.time()
            row = _gate_one(name, K, how, y_tr, Xfix_tr, Kbase_tr, labels_tr,
                            cfg, base_oof, base_folds, cfg.SEED + gi)
            rows.append(row)
            if ckpt_cfg is not None:
                ck_save(ckpt_cfg, f"gate_slice_{name}", fp, row)
            print(f"      -> dPCC={row['dPCC']:+.4f}  ({time.time()-_t0:.0f}s)", flush=True)
    g = pd.DataFrame(rows)
    if len(g) == 0:
        return g
    g["q_BH"] = bh_fdr(g["p_boot"].values)
    g["sig_positive"] = (g["CI_low"] > 0) & (g["q_BH"] < cfg.FDR_Q)
    g["stable"] = g["stability_cv"] < cfg.STABILITY_MAX_CV
    g["PASS"] = g["sig_positive"] & g["stable"]
    # ---- classification fixed in advance, to prevent selective reporting ----
    def classify(r):
        if not r["PASS"]:
            return "negative (evidence for a diffuse architecture)"
        if np.isfinite(global_dpcc) and r["dPCC"] > global_dpcc:
            return "concentrated candidate (goes to deep cis scanning)"
        return "compatible with diffuse (the aggregate layer still dominates)"
    g["classification"] = g.apply(classify, axis=1)
    g["global_dPCC_ref"] = global_dpcc
    return g


# =====================================================================
#  Part 2  Search spaces on two tracks. Track 2 must never use the phenotype y.
#     track 1 (pool-based)     AA / AD / MULT: marginal pool x cross-chromosome,
#                              plus near neighbours of the pool anchors
#     track 2 (cis, no pool)   DD / threshold: exhaustive same-chromosome window
#     track 2 (trans, no pool) XOR: cross-chromosome sample after a MAF and
#                              heterozygosity filter
#     trans DD / threshold     not scanned pair by pair; that resolution is
#                              left to the aggregate kernel
# =====================================================================
def build_pool_pairs(pool, chrom, pos, cfg):
    """Track 1: cross-chromosome pairs within the pool, plus same-chromosome neighbours of pool anchors."""
    pool = np.asarray(pool)
    cross = set()
    for a in range(len(pool)):
        for b in range(a + 1, len(pool)):
            i, j = pool[a], pool[b]
            if chrom[i] != chrom[j]:
                cross.add((int(min(i, j)), int(max(i, j))))
    win = cfg.NEAR_WINDOW_KB * 1000
    near = set()
    for i in pool:
        same = np.where(chrom == chrom[i])[0]
        d = np.abs(pos[same] - pos[i])
        nb = same[(d > 0) & (d <= win)]
        if len(nb) > cfg.MAX_NEAR_PER_ANCHOR:
            nb = nb[np.argsort(d[(d > 0) & (d <= win)])[:cfg.MAX_NEAR_PER_ANCHOR]]
        for j in nb:
            near.add((int(min(i, j)), int(max(i, j))))
    return sorted(cross), sorted(near - cross)


def build_cis_pairs(chrom, pos, cfg):
    """Track 2 (cis): exhaustive same-chromosome pairs within CIS_WINDOW_MB. No pool, no phenotype."""
    return cis_pair_list(chrom, pos, int(cfg.CIS_WINDOW_MB * 1e6), cfg.MAX_CIS_NEIGHBORS)


def build_xor_pairs(X, chrom, af1, cfg):
    """Track 2 (trans): sampled cross-chromosome XOR pairs after MAF and heterozygosity filtering. No phenotype."""
    rng = np.random.RandomState(cfg.SEED)
    xpool = xor_eligible(X, af1, cfg)
    if len(xpool) < 2:
        return []
    # total number of cross-chromosome pairs available
    tot = 0
    for c in np.unique(chrom[xpool]):
        n_c = int((chrom[xpool] == c).sum())
        tot += n_c * (len(xpool) - n_c)
    tot //= 2
    if tot <= cfg.XOR_MAX_PAIRS:
        prs = set()
        for a in range(len(xpool)):
            for b in range(a + 1, len(xpool)):
                i, j = xpool[a], xpool[b]
                if chrom[i] != chrom[j]:
                    prs.add((int(min(i, j)), int(max(i, j))))
        return sorted(prs)
    return trans_pair_sample(chrom, cfg.XOR_MAX_PAIRS, rng, snp_subset=xpool)


# =====================================================================
#  Control for phantom cis epistasis (Wood et al. 2014)
#  Two nearby markers can jointly tag a causal variant that neither tags
#  additively on its own, so their apparent DD or threshold interaction may be
#  nothing but a non-linearity in local haplotype dose. Conditioning on the two
#  additive main effects does NOT remove it. The control adds local haplotype
#  information (PCs of the SNPs in the surrounding window) to the null model as
#  extra fixed effects and recomputes F; only a pair that keeps its signal
#  remains a candidate.
# =====================================================================
def local_window_pcs(X, chrom, pos, i, j, window_bp, n_pc=3, exclude=(), max_snp=200):
    lo, hi = min(pos[i], pos[j]) - window_bp, max(pos[i], pos[j]) + window_bp
    sel = np.where((chrom == chrom[i]) & (pos >= lo) & (pos <= hi))[0]
    sel = np.array([s for s in sel if s not in exclude])
    if len(sel) < 2:
        return []
    if len(sel) > max_snp:
        sel = sel[np.linspace(0, len(sel) - 1, max_snp).astype(int)]
    Z = X[:, sel].astype(float)
    Z = Z - Z.mean(0)
    sd = Z.std(0); sd[sd < 1e-12] = 1.0
    Z = Z / sd
    try:
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:
        return []
    k = min(n_pc, U.shape[1])
    return [U[:, c] for c in range(k)]


def cis_haplotype_control(df_cis, X, Ha, Hd, chrom, pos, y, af1, global_mains, cfg, top_n=200):
    """Haplotype control for cis hits: recompute F with local window PCs added. Reports F_hapctrl and survival."""
    if len(df_cis) == 0:
        return df_cis
    df = df_cis.copy()
    for col in ["F_DD", "F_THRESH"]:
        if col not in df.columns:
            df[col] = 0.0
    df["_maxF"] = df[["F_DD", "F_THRESH"]].fillna(0).max(axis=1)
    cand = df.sort_values("_maxF", ascending=False).head(top_n).index
    # object dtype holds the booleans; pandas 3.0 refuses True in a float64 column
    df["F_hapctrl"] = np.nan
    df["survives_haplotype_control"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    win = int(cfg.CIS_WINDOW_MB * 1e6)
    for ix in cand:
        i = int(df.at[ix, "snp_i"]); j = int(df.at[ix, "snp_j"])
        pcs = local_window_pcs(X, chrom, pos, i, j, win, cfg.HAP_CTRL_PCS, exclude=(i, j))
        mains = list(global_mains) + [Ha[:, i], Hd[:, i], Ha[:, j], Hd[:, j]] + pcs
        base = float(df.at[ix, "_maxF"])
        if base <= 0:
            continue
        if float(df.at[ix, "F_DD"] or 0) >= float(df.at[ix, "F_THRESH"] or 0):
            inter = Hd[:, i] * Hd[:, j]
        else:
            md = np.where(af1[None, :] < 0.5, X, 2 - X)
            k = df.at[ix, "best_k_THRESH"]
            k = int(k) if np.isfinite(k) else cfg.KERNEL_THRESH_K
            inter = ((md[:, i] + md[:, j]) >= k).astype(float)
        inter = inter - inter.mean()
        if inter.std() < 1e-12:
            continue
        Fh = interaction_F_conditioned(y, inter, mains)
        df.at[ix, "F_hapctrl"] = float(Fh)
        df.at[ix, "survives_haplotype_control"] = bool(Fh > cfg.HAP_CTRL_KEEP_FRAC * base)
    return df.drop(columns=["_maxF"])


# =====================================================================
#  Part 3  Pair scoring, by track. No threshold filter; everything is written out.
# =====================================================================
def _score_chunk_typed(pairs, y, Xa, Ha, Hd, af1, global_mains, cfg, types):
    out = []
    for (i, j) in pairs:
        r = score_pair(i, j, y, Xa, Ha, Hd, af1, global_mains,
                       min_count=cfg.MIN_CELL_COUNT_EFF,
                       thr_ks=cfg.THRESHOLD_K_RANGE, types_to_compute=types)
        out.append(r)
    return out


def _safe_n_jobs(cfg, n_ind, n_snp, verbose=True):
    """
    joblib copies Xa / Ha / Hd into every worker process. At n=2796, m=35750
    those three float32 matrices are about 1.2GB, so 48 workers need about
    57GB; with the kernels in the parent process that exhausts memory, usually
    only after hours of running. The worker count is therefore capped by the
    free memory.
    """
    req = cfg.N_JOBS
    nj = cpu_count() if req in (-1, None) else int(req)
    per_proc_gb = 3 * n_ind * n_snp * 4 / 1e9          # three float32 matrices
    avail_gb = avail_mem_gb()
    budget = max(1, int((avail_gb * cfg.MEM_FRACTION) / max(per_proc_gb, 1e-9)))
    nj_safe = max(1, min(nj, budget, cpu_count()))
    if verbose and nj_safe < nj:
        print(f"    [memory guard] about {per_proc_gb:.1f}GB per process, "
              f"{avail_gb:.0f}GB free => workers reduced from {nj} to {nj_safe}")
    return nj_safe


def run_scoring_typed(pairs, y, Xa, Ha, Hd, af1, global_mains, cfg, types, tag):
    if len(pairs) == 0:
        return pd.DataFrame()
    t0 = time.time()
    if _HAS_JOBLIB and cfg.N_JOBS != 1 and len(pairs) > 2000:
        nj = _safe_n_jobs(cfg, Xa.shape[0], Xa.shape[1])
        chunks = np.array_split(np.array(pairs, dtype=int), max(nj * 4, 1))
        with par_ctx(1):
            res = Parallel(n_jobs=nj, backend="loky")(
                delayed(_score_chunk_typed)([tuple(map(int, p)) for p in ch], y, Xa, Ha, Hd,
                                            af1, global_mains, cfg, types)
                for ch in chunks if len(ch))
        rows = [r for ch in res for r in ch]
    else:
        rows = _score_chunk_typed([tuple(map(int, p)) for p in pairs], y, Xa, Ha, Hd,
                                  af1, global_mains, cfg, types)
    df = pd.DataFrame(rows)
    df["search_track"] = tag
    print(f"    {tag}: {len(df)} pairs ({time.time()-t0:.1f}s)")
    return df


# =====================================================================
#  Checkpointing
#  A cache entry is bound to a fingerprint of the input data and the parameters
#  that affect it. Change the trait, the window or the split seed and the
#  fingerprint changes, so a stale intermediate can never be fed to a run it
#  does not belong to. That failure mode is silent, and therefore far more
#  dangerous than a crash.
# =====================================================================
# Entries depending only on genotypes, positions and structure parameters are
# trait-independent and can be shared across traits.
GENO_ONLY_CACHES = {"kernels", "structure", "slice_kernels"}


def _ckpt_dir(cfg, name=None):
    """
    Trait-independent artefacts (GRM, structure, slice kernels) go to the
    shared SHARED_CKPT_DIR; trait-dependent ones (Part 1, the gate, scoring)
    go to OUT_DIR/_ckpt of that trait, so switching trait neither rebuilds the
    kernels nor overwrites another trait's results.
    """
    if name in GENO_ONLY_CACHES and getattr(cfg, "SHARED_CKPT_DIR", None):
        d = cfg.SHARED_CKPT_DIR
    else:
        d = cfg.CKPT_DIR or os.path.join(cfg.OUT_DIR, "_ckpt")
    os.makedirs(d, exist_ok=True)
    return d


def data_fingerprint(X, y, chrom, pos, cfg):
    """Fingerprint of the input data plus the parameters that affect intermediates."""
    h = hashlib.md5()
    h.update(np.ascontiguousarray(X[:, :min(200, X.shape[1])]).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    h.update(str(X.shape).encode())
    h.update("|".join(map(str, chrom[:200])).encode())
    h.update(np.ascontiguousarray(pos[:200]).tobytes())
    for k in ["TRAIT", "SEED", "TEST_FRAC", "N_CV_FOLDS", "N_PCS", "CIS_WINDOW_MB",
              "NEAR_WINDOW_KB", "POOL_SIZE", "MAX_CIS_NEIGHBORS", "KERNEL_TRANS_SAMPLE",
              "KERNEL_THRESH_K", "XOR_MIN_MAF", "XOR_MIN_HET", "XOR_MAX_PAIRS",
              "MIN_CELL_COUNT_EFF", "REML_MAXITER", "REML_RESTARTS", "N_BOOT",
              "GATE_REML_MAXITER", "MAX_MAJOR_SNPS", "MAX_NEAR_PER_ANCHOR"]:
        h.update(f"{k}={getattr(cfg, k, None)}".encode())
    return h.hexdigest()[:16]


def geno_fingerprint(X, chrom, pos, cfg):
    """Genotypes, positions and structure parameters only -- no phenotype and no
    trait name. The GRM and the slice kernels do not depend on the trait, so
    this fingerprint lets them be shared across traits. If a trait drops
    different individuals, X changes, the fingerprint changes and the cache is
    invalidated automatically.
    """
    h = hashlib.md5()
    h.update(np.ascontiguousarray(X[:, :min(200, X.shape[1])]).tobytes())
    h.update(str(X.shape).encode())
    h.update("|".join(map(str, chrom[:200])).encode())
    h.update(np.ascontiguousarray(pos[:200]).tobytes())
    for k in ["SEED", "TEST_FRAC", "N_PCS", "CIS_WINDOW_MB", "MAX_CIS_NEIGHBORS",
              "KERNEL_TRANS_SAMPLE", "KERNEL_THRESH_K", "XOR_MIN_MAF", "XOR_MIN_HET",
              "FALLBACK_MAX_K", "USE_ADMIXTURE"]:
        h.update(f"{k}={getattr(cfg, k, None)}".encode())
    return h.hexdigest()[:16]


def ck_load(cfg, name, fp):
    """Load a cache entry; returns None if the fingerprint differs or RESUME is off."""
    if not getattr(cfg, "RESUME", False):
        return None
    p = os.path.join(_ckpt_dir(cfg, name), f"{name}.pkl")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if obj.get("_fp") != fp:
            print(f"    [ckpt] {name}: fingerprint differs (data or parameters changed); recomputing")
            return None
        print(f"    [ckpt] reusing completed stage: {name}")
        return obj["data"]
    except Exception as e:
        print(f"    [ckpt] {name} could not be read ({e}); recomputing")
        return None


def ck_save(cfg, name, fp, data):
    if not getattr(cfg, "RESUME", False):
        return
    p = os.path.join(_ckpt_dir(cfg, name), f"{name}.pkl")
    tmp = p + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump({"_fp": fp, "data": data}, f, protocol=4)
        os.replace(tmp, p)     # atomic: an interrupt cannot leave a half-written file
    except Exception as e:
        print(f"    [ckpt] {name} could not be saved ({e}); computation continues")


# =====================================================================
def build_kernels(X):
    Ha, Hd = noia(X)
    K_A = grm_a(X); K_D = grm_d(Hd)
    return {"A": K_A, "D": K_D,
            "AA": hadamard(K_A, K_A),
            "DD": hadamard(K_D, K_D),
            "AD": hadamard(K_A, K_D)}, Ha, Hd


# =====================================================================
#  Main driver
# =====================================================================
def run_pipeline(X, y, chrom, pos, snp_names, cfg, extra_cov=None):
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    n, m = X.shape
    t0 = time.time()
    y = (y - y.mean()) / (y.std() + 1e-12)

    print("=" * 74)
    print(f"[Part 0] data n={n}, SNP={m}, trait={cfg.TRAIT}")
    print("=" * 74)
    fp = data_fingerprint(X, y, chrom, pos, cfg)      # trait-dependent stages
    gfp = geno_fingerprint(X, chrom, pos, cfg)        # trait-independent stages
    if getattr(cfg, "RESUME", False):
        print(f"  checkpointing on; cache dir {_ckpt_dir(cfg)}  fingerprint={fp}")

    cached = ck_load(cfg, "kernels", gfp)
    if cached is None:
        print("  building GRMs (A/D/AA/DD/AD)...")
        Kdict, Ha, Hd = build_kernels(X)
        ck_save(cfg, "kernels", gfp, (Kdict, Ha, Hd))
    else:
        Kdict, Ha, Hd = cached
    K_A = Kdict["A"]
    af1 = X.mean(0) / 2.0

    cached = ck_load(cfg, "structure", gfp)
    if cached is None:
        admix_cv, admix_out = (None, None)
        if getattr(cfg, "USE_ADMIXTURE", True) and os.path.exists(cfg.BFILE + ".bed"):
            print(f"  running ADMIXTURE (K={list(cfg.ADMIX_K_RANGE)}, "
                  f"-j{cfg.ADMIX_THREADS}). This can be slow; the result is cached.")
            admix_cv, admix_out = run_admixture_cv(
                cfg.BFILE, cfg.ADMIX_K_RANGE, os.path.join(cfg.OUT_DIR, "admix"),
                cfg.PLINK_BIN, cfg.ADMIX_BIN, cfg.ADMIX_THREADS)
        elif not getattr(cfg, "USE_ADMIXTURE", True):
            print("  USE_ADMIXTURE=False: using GRM-PCA + KMeans instead (seconds)")
        labels, struct_meta = structure_clusters(K_A, admix_out, cfg.FALLBACK_MAX_K, cfg.SEED)
        is_test = stratified_split(labels, cfg.TEST_FRAC, cfg.SEED)
        ck_save(cfg, "structure", gfp, (labels, struct_meta, admix_cv, is_test))
    else:
        labels, struct_meta, admix_cv, is_test = cached
    n_clusters = len(np.unique(labels))
    print(f"  population structure: method={struct_meta[0]}, clusters={n_clusters}")

    tr_idx = np.where(~is_test)[0]; te_idx = np.where(is_test)[0]
    labels_tr = labels[tr_idx]
    print(f"  80:20 stratified split: train {len(tr_idx)} / test {len(te_idx)} (test frozen)")

    PC_all = pca_from_grm(K_A, cfg.N_PCS)
    fixed_cols = [np.ones(n), *[PC_all[:, c] for c in range(PC_all.shape[1])]]
    if extra_cov is not None and extra_cov.shape[1] > 0:
        fixed_cols += [extra_cov[:, c] for c in range(extra_cov.shape[1])]
    Xfix = np.column_stack(fixed_cols)

    # ---------------- Part 1: is the aggregate layer usable at all ----------------
    print("\n" + "=" * 74)
    print(f"[Part 1] aggregate layer: {cfg.N_CV_FOLDS}-fold CV inside the training "
          f"set plus the frozen test set")
    print("=" * 74)
    y_tr = y[tr_idx]; Xfix_tr = Xfix[tr_idx]
    Ktr = {k: _sub(Kdict[k], tr_idx, tr_idx) for k in Kdict}
    cached = ck_load(cfg, "part1_cv", fp)
    if cached is None:
        cv_df, cv_summ, delta_df, delta_summ, prop_acc = run_cv_prediction(
            y_tr, Xfix_tr, Ktr, labels_tr, cfg)
        ck_save(cfg, "part1_cv", fp, (cv_df, cv_summ, delta_df, delta_summ, prop_acc))
    else:
        cv_df, cv_summ, delta_df, delta_summ, prop_acc = cached
    print("\n  CV prediction PCC per model (mean +/- SD):")
    print(cv_summ.round(4).to_string(index=False))
    print("\n  dPCC (per-fold mean +/- SD):")
    print(delta_summ.round(4).to_string(index=False))
    epi_row = delta_summ[delta_summ["contrast"].str.contains("epistasis")].iloc[0]
    global_dpcc = float(epi_row["dPCC_mean"])
    print(f"\n  >>> epistatic gain of the global aggregate kernel, "
          f"dPCC(A+D+AA - A+D) = "
          f"{global_dpcc:+.4f} ± {epi_row['dPCC_sd']:.4f}")

    cached = ck_load(cfg, "part1_holdout", fp)
    if cached is None:
        hold_df, hold_delta, props, logls = run_holdout_prediction(
            y, Xfix, Kdict, tr_idx, te_idx, cfg)
        ck_save(cfg, "part1_holdout", fp, (hold_df, hold_delta, props, logls))
    else:
        hold_df, hold_delta, props, logls = cached
    print("\n  held-out test set dPCC with paired-bootstrap 95% CI:")
    print(hold_delta.round(4).to_string(index=False))

    # balanced evaluation metrics (continuous trait)
    met_rows = ck_load(cfg, "part1_metrics", fp) or []
    for mname, keys in ([] if met_rows else PRED_MODELS.items()):
        try:
            pred, _, _ = predict_fold(y, Xfix, Kdict, tr_idx, te_idx, keys)
            r = eval_metrics(pred, y[te_idx], cfg.TOP_FRAC, cfg.OVERLAP_FRAC)
            r["model"] = mname; met_rows.append(r)
        except Exception:
            pass
    ck_save(cfg, "part1_metrics", fp, met_rows)
    met_df = pd.DataFrame(met_rows)
    if len(met_df):
        met_df = met_df[["model", "PCC", "R2", "Spearman", "top_gain",
                         "top_overlap", "calib_slope"]]
        print("\n  balanced evaluation on the frozen test set:")
        print(met_df.round(4).to_string(index=False))

    vc_rows = []
    for m_, keys in PRED_MODELS.items():
        row = {"model": m_}
        for k, val in zip(keys + ["resid"], props[m_]):
            row[f"prop_{k}"] = val
        vc_rows.append(row)
    vc_df = pd.DataFrame(vc_rows)

    # ---------------- PART1_ONLY: fast trait screen, no slice kernels ----------------
    if getattr(cfg, "PART1_ONLY", False):
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        O = cfg.OUT_DIR
        cv_summ.to_csv(os.path.join(O, "prediction_cv.csv"), index=False)
        delta_summ.to_csv(os.path.join(O, "prediction_cv_delta.csv"), index=False)
        hold_df.to_csv(os.path.join(O, "prediction_holdout.csv"), index=False)
        hold_delta.to_csv(os.path.join(O, "prediction_holdout_delta.csv"), index=False)
        if len(met_df):
            met_df.to_csv(os.path.join(O, "evaluation_metrics.csv"), index=False)
        vc_df.to_csv(os.path.join(O, "variance_components.csv"), index=False)
        hd = hold_delta[hold_delta["contrast"].str.startswith(EPI_MODEL)]
        summ = {"trait": cfg.TRAIT, "n": int(n), "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
                "dPCC_cv_mean": global_dpcc,
                "dPCC_cv_sd": float(epi_row["dPCC_sd"]),
                "dPCC_holdout": float(hd["dPCC"].iloc[0]) if len(hd) else None,
                "CI_low": float(hd["CI_low"].iloc[0]) if len(hd) else None,
                "CI_high": float(hd["CI_high"].iloc[0]) if len(hd) else None,
                "excludes_0": bool(hd["excludes_0"].iloc[0]) if len(hd) else None,
                "runtime_min": round((time.time() - t0) / 60, 2)}
        json.dump(summ, open(os.path.join(O, "part1_summary.json"), "w"),
                  ensure_ascii=False, indent=2)
        print(f"\n  [PART1_ONLY] Part 1 results written to {O}/; slice kernels "
              f"and pair scanning skipped")
        print(f"  >>> {cfg.TRAIT}: CV ΔPCC={global_dpcc:+.4f}±{epi_row['dPCC_sd']:.4f}  "
              f"holdout ΔPCC={summ['dPCC_holdout']:+.4f} "
              f"CI[{summ['CI_low']:+.4f},{summ['CI_high']:+.4f}] "
              f"{'excludes 0' if summ['excludes_0'] else 'includes 0'}")
        return summ, pd.DataFrame(), pd.DataFrame()

    # ---------------- Part 1.5: slice kernels and the gate ----------------
    print("\n" + "=" * 74)
    print("[Part 1.5] multi-resolution slice kernels and the gate")
    print("=" * 74)
    print("  Slice kernels are built from genotypes and positions only, never "
          "from the phenotype, so leakage is structurally impossible.")
    cached = ck_load(cfg, "slice_kernels", gfp)
    if cached is None:
        slice_K = build_slice_kernels(X, Ha, Hd, chrom, pos, af1, cfg)
        ck_save(cfg, "slice_kernels", gfp, slice_K)
    else:
        slice_K = cached
    slice_K_tr = {k: (_sub(K, tr_idx, tr_idx), how) for k, (K, how) in slice_K.items()}
    Kbase_tr = {"A": Ktr["A"], "D": Ktr["D"]}
    gate = run_gating(y_tr, Xfix_tr, Kbase_tr, slice_K_tr, labels_tr, cfg, global_dpcc,
                      ckpt_cfg=cfg, fp=fp)
    if len(gate):
        print(gate[["slice", "dPCC", "CI_low", "CI_high", "q_BH",
                    "stability_cv", "PASS", "classification"]].round(4).to_string(index=False))
        n_pass = int(gate["PASS"].sum())
        print(f"\n  {n_pass}/{len(gate)} slice kernel(s) passed the gate.")
        if n_pass == 0:
            print("  >>> All negative: concentrated signal was actively looked for at")
            print("      several resolutions and was not found. That is positive\n"
                  "      evidence for a diffuse architecture.")

    # trans aggregate field (per-individual score = kernel diagonal): usable for
    # ranking individuals and for allocation, but it carries no locus information
    field_rows = {"IID_index": np.arange(n)}
    for name, (K, _) in slice_K.items():
        if name.startswith("trans"):
            passed = bool(len(gate) and gate.loc[gate["slice"] == name, "PASS"].any())
            field_rows[f"field_{name}" + ("_PASS" if passed else "")] = np.diag(K)
    field_df = pd.DataFrame(field_rows)

    # ---------------- pre-registered gate on whether to scan at all ----------------
    # Track A is triggered by cis-AA or trans-AA passing the gate; track B by
    # cis-DD, cis-threshold or cis-XOR passing it. If no kernel passes, neither
    # track is triggered and no pair scanning is done.
    # This is not only about cost. Without gate support, the top-scoring pair
    # among millions is almost certainly noise -- a single pair needs about 2% of
    # phenotypic variance to be detectable here and the strongest real ones are
    # around 0.01%. Reporting it is exactly the post-hoc fishing this design
    # exists to prevent.
    n_pass = int(gate["PASS"].sum()) if len(gate) else 0
    skip_scan = bool(getattr(cfg, "RESPECT_GATE", True) and n_pass == 0)
    logp = np.zeros(m); major = np.array([], dtype=int); pool = np.array([], dtype=int)
    thr = np.nan
    all_score = pd.DataFrame()

    if skip_scan:
        print("\n" + "=" * 74)
        print("[Part 2/3] pair scanning skipped by the pre-registered rule")
        print("=" * 74)
        print("  No slice kernel passed the gate, so neither track was triggered.")
        print("  Without gate support the top-scoring pair at this sample size is")
        print("  almost certainly noise, and reporting it would be exactly the kind")
        print("  of post-hoc fishing this design exists to prevent.\n"
              "  (To force a scan anyway: cfg.RESPECT_GATE=False or --force_scan)")
    else:
        # ---------------- Part 2: search spaces ----------------
        print("\n" + "=" * 74)
        print("[Part 2] main-effect conditioning and the two search spaces "
              "(track 2 never uses the phenotype)")
        print("=" * 74)
        X_tr = X[tr_idx]; Ha_tr, Hd_tr = Ha[tr_idx], Hd[tr_idx]
        cached = ck_load(cfg, "part2_spaces", fp)
        if cached is not None:
            logp, major, thr, pairs_cross, pairs_near, cis_pairs, xor_pairs, pool = cached
        else:
            logp, _ = quick_additive_gwas(X_tr, y_tr, Xfix_tr)
            major, thr = pick_major_effect_snps(logp, snp_names, X_tr, cfg.MAX_MAJOR_SNPS)
            pool = np.argsort(-logp)[:cfg.POOL_SIZE]
            pairs_cross, pairs_near = build_pool_pairs(pool, chrom, pos, cfg)
            cis_pairs = build_cis_pairs(chrom, pos, cfg)
            xor_pairs = build_xor_pairs(X_tr, chrom, af1, cfg)
            ck_save(cfg, "part2_spaces", fp,
                    (logp, major, thr, pairs_cross, pairs_near, cis_pairs, xor_pairs, pool))
        print(f"  additive GWAS: Bonferroni -log10p threshold {thr:.2f}, "
              f"conditioning on {len(major)} main locus/loci")
        global_mains = [Xfix_tr[:, c] for c in range(1, Xfix_tr.shape[1])] + \
                       [Ha_tr[:, k] for k in major]
        print(f"  track 1 (pool): {len(pairs_cross)} cross-chromosome pairs, "
              f"{len(pairs_near)} near pairs")
        print(f"  track 2 (cis, exhaustive): {len(cis_pairs)} pairs")
        print(f"  track 2 (trans, XOR sample): {len(xor_pairs)} pairs")

        # ---------------- Part 3: pair scoring ----------------
        print("\n" + "=" * 74)
        print("[Part 3] pair scoring (everything is written out; no filtering)")
        print("=" * 74)
        parts = []
        def _scored(tag, prs, types):
            c = ck_load(cfg, f"score_{tag}", fp)
            if c is not None:
                return c
            d = run_scoring_typed(prs, y_tr, X_tr, Ha_tr, Hd_tr, af1, global_mains,
                                  cfg, types, tag)
            ck_save(cfg, f"score_{tag}", fp, d)
            return d
        d1 = _scored("pool_cross", pairs_cross, ["AA", "AD", "MULT"])
        d2 = _scored("pool_near", pairs_near, ["AA", "AD", "MULT"])
        d3 = _scored("cis_DD_threshold", cis_pairs, ["DD", "THRESH"])
        d4 = _scored("xor_trans", xor_pairs, ["XOR"])
        # phantom-epistasis control for cis hits
        if len(d3) and "F_hapctrl" not in d3.columns:
            d3 = cis_haplotype_control(d3, X_tr, Ha_tr, Hd_tr, chrom, pos, y_tr, af1,
                                       global_mains, cfg, cfg.HAP_CTRL_TOP_N)
            ck_save(cfg, "score_cis_DD_threshold", fp, d3)
        if len(d3) and "survives_haplotype_control" in d3.columns:
            nsurv = int((d3["survives_haplotype_control"] == True).sum())
            print(f"    cis haplotype control: {int(d3['F_hapctrl'].notna().sum())} "
                  f"pair(s) tested, {nsurv} survived")
        for d in (d1, d2, d3, d4):
            if len(d):
                parts.append(d)
        all_score = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if len(all_score):
        all_score["snp_i_name"] = snp_names[all_score["snp_i"].values]
        all_score["snp_j_name"] = snp_names[all_score["snp_j"].values]
        all_score["chrom_i"] = chrom[all_score["snp_i"].values]
        all_score["chrom_j"] = chrom[all_score["snp_j"].values]
        all_score["pos_i"] = pos[all_score["snp_i"].values]
        all_score["pos_j"] = pos[all_score["snp_j"].values]
        all_score["same_chrom"] = all_score["chrom_i"] == all_score["chrom_j"]
        all_score["dist_bp"] = np.where(all_score["same_chrom"],
                                        np.abs(all_score["pos_i"] - all_score["pos_j"]), -1)

    # ---------------- output ----------------
    print("\n" + "=" * 74)
    print("[output] writing files")
    print("=" * 74)
    O = cfg.OUT_DIR
    pd.DataFrame({"index": np.arange(n), "cluster": labels,
                  "split": np.where(is_test, "test", "train")}).to_csv(
        os.path.join(O, "split_assignment.csv"), index=False)
    json.dump({"method": struct_meta[0], "n_clusters": int(n_clusters),
               "admix_cv": admix_cv, "detail": str(struct_meta[1])},
              open(os.path.join(O, "structure_info.json"), "w"),
              ensure_ascii=False, indent=2)
    cv_df.to_csv(os.path.join(O, "prediction_cv_perfold.csv"), index=False)
    cv_summ.to_csv(os.path.join(O, "prediction_cv.csv"), index=False)
    delta_summ.to_csv(os.path.join(O, "prediction_cv_delta.csv"), index=False)
    hold_df.to_csv(os.path.join(O, "prediction_holdout.csv"), index=False)
    hold_delta.to_csv(os.path.join(O, "prediction_holdout_delta.csv"), index=False)
    if len(met_df):
        met_df.to_csv(os.path.join(O, "evaluation_metrics.csv"), index=False)
    vc_df.to_csv(os.path.join(O, "variance_components.csv"), index=False)
    if len(gate):
        gate.to_csv(os.path.join(O, "slice_kernel_gating.csv"), index=False)
    field_df.to_csv(os.path.join(O, "aggregate_field_scores.csv"), index=False)
    pd.DataFrame({"snp": snp_names[major], "index": major}).to_csv(
        os.path.join(O, "conditioned_major_snps.csv"), index=False)
    pd.DataFrame({"snp": snp_names[pool], "index": pool,
                  "logp": logp[pool]}).to_csv(os.path.join(O, "candidate_pool.csv"), index=False)
    if len(all_score):
        all_score.to_csv(os.path.join(O, "epistasis_scores_all.csv"), index=False)

    summary = {
        "n": int(n), "n_snp": int(m), "trait": cfg.TRAIT,
        "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
        "n_clusters": int(n_clusters), "structure_method": struct_meta[0],
        "global_epistasis_dPCC_cv": global_dpcc,
        "holdout": hold_delta.to_dict("records"),
        "n_slices": int(len(gate)), "n_slices_pass": int(gate["PASS"].sum()) if len(gate) else 0,
        "slice_classifications": (gate.set_index("slice")["classification"].to_dict()
                                  if len(gate) else {}),
        "n_pairs_scored": int(len(all_score)),
        "gate_n_pass": int(n_pass),
        "pairwise_scan_skipped": bool(skip_scan),
        "skip_reason": ("no slice kernel passed the gate; no track triggered"
                        if skip_scan else ""),
        "cis_window_Mb": cfg.CIS_WINDOW_MB, "near_window_kb": cfg.NEAR_WINDOW_KB,
        "min_cell_count": cfg.MIN_CELL_COUNT_EFF,
        "runtime_min": round((time.time() - t0) / 60, 2),
        "NOTE": ("Variance fractions are diagnostic only; conclusions rest on the "
                 "held-out dPCC. Slice kernels never use the phenotype. The gate "
                 "is an attempt to falsify the diffuse hypothesis, so an all-"
                 "negative gate is evidence for it."),
    }
    json.dump(summary, open(os.path.join(O, "run_summary.json"), "w"),
              ensure_ascii=False, indent=2, default=str)
    print(f"  all results written to {O}/   total runtime "
          f"{summary['runtime_min']} min")
    return summary, all_score, gate


# =====================================================================
#  Loading real data
# =====================================================================
def load_pheno_gcta(path, iids, trait):
    """GCTA long format: FID IID VALUE, no header."""
    d = pd.read_csv(path, sep=r"\s+", header=None, engine="python",
                    names=["FID", "IID", trait])
    d["IID"] = d["IID"].astype(str)
    d = d.drop_duplicates("IID").set_index("IID")
    return d.reindex(iids)[trait].values.astype(float)


def load_covs_gcta(qual_path, quant_path, iids):
    """
    Covariate reading:
      categorical (.cov)  cast to string, then get_dummies(drop_first=True)
      continuous (.qcov)  numeric as-is
      both blocks are concatenated and reindexed to the IID order of the .fam
    Returns a DataFrame that may contain NaN. The caller DROPS those
    individuals; no mean imputation is performed.
    """
    parts = []
    if qual_path and os.path.exists(qual_path):
        q = pd.read_csv(qual_path, sep=r"\s+", header=None, engine="python")
        q.columns = ["FID", "IID"] + [f"q{i}" for i in range(q.shape[1] - 2)]
        q["IID"] = q["IID"].astype(str)
        du = pd.get_dummies(
            q.set_index("IID")[[c for c in q.columns if c.startswith("q")]].astype(str),
            drop_first=True).astype(float)
        parts.append(du.reindex(iids))
    if quant_path and os.path.exists(quant_path):
        z = pd.read_csv(quant_path, sep=r"\s+", header=None, engine="python")
        z.columns = ["FID", "IID"] + [f"z{i}" for i in range(z.shape[1] - 2)]
        z["IID"] = z["IID"].astype(str)
        parts.append(z.set_index("IID")[[c for c in z.columns
                                         if c.startswith("z")]].astype(float).reindex(iids))
    if not parts:
        return None
    return pd.concat(parts, axis=1)


def load_real(cfg):
    """
    Read .bed/.bim/.fam together with GCTA phenotype and covariate files.

      (1) individuals are matched on the raw IID string from the .fam file,
          with no normalisation or rewriting;
      (2) an individual with any missing covariate is dropped entirely and never
          mean-imputed: imputing a categorical covariate would invent a level
          that does not exist;
      (3) whether covariates entered the model is printed explicitly, so a run
          cannot silently become a no-covariate run.
    """
    bim = read_bim(cfg.BFILE); fam = read_fam(cfg.BFILE)
    n_all, m = len(fam), len(bim)
    print(f"  read {cfg.BFILE}: n={n_all}, SNP={m}")
    ids = fam["iid"].astype(str).values

    yv = load_pheno_gcta(cfg.PHENO_FILE, ids, cfg.TRAIT)
    ok = np.isfinite(yv)
    print(f"  phenotype {cfg.PHENO_FILE}: matched {int(ok.sum())}/{n_all}")
    if ok.sum() < 50:
        raise SystemExit(
            f"too few phenotyped individuals matched ({int(ok.sum())}). Check that "
            f"the .fam IID and column 2 of {cfg.PHENO_FILE} use the same identifiers.")

    covM = load_covs_gcta(cfg.COV_FILE, cfg.QCOV_FILE, ids)
    if covM is None:
        print("  no covariate file given (--cov/--qcov): the model contains NO "
              "covariates, only an intercept and the structure PCs.")
    else:
        bad = covM.isnull().any(axis=1).values
        ok = ok & (~bad)
        print(f"  covariates: {covM.shape[1]} column(s) "
              f"(categorical dummy-coded, drop_first=True); "
              f"{int(bad.sum())} individual(s) dropped for missing covariates")

    keep = np.where(ok)[0]
    print(f"  >>> individuals entering the analysis: {len(keep)}")

    X = read_bed(cfg.BFILE, n_all, m)[keep]
    X = np.ascontiguousarray(X, dtype=np.float32)   # halves memory (2796x35750: 800MB -> 400MB)
    yv = yv[keep]
    extra = covM.values[keep].astype(float) if covM is not None else None

    # missing genotypes (-1) are filled with the marker mean
    miss = (X < 0)
    if miss.any():
        for c in np.where(miss.any(0))[0]:
            col = X[:, c]
            obs = col[col >= 0]
            col[col < 0] = obs.mean() if len(obs) else 0.0
            X[:, c] = col

    chrom = bim["chrom"].astype(str).values
    pos = bim["pos"].astype(np.int64).values
    snp_names = bim["snp"].astype(str).values
    return X, yv, chrom, pos, snp_names, extra


# =====================================================================
#  Self-test
# =====================================================================
def synth_data(seed=1, n=900, per_chr=130):
    """
    Simulate structured data containing both ends of the architecture:
      (i)  a polygenic AA background, which the aggregate kernel can predict
           but no single pair can show;
      (ii) five concentrated true interaction pairs (AA/AD/DD/threshold/XOR),
           which the pairwise tests should score highly.
    Variance shares: additive 0.15, polygenic AA 0.25, concentrated
    interactions 0.20, noise 0.40.
    """
    rng = np.random.RandomState(seed)
    n_chr = 6; m = n_chr * per_chr
    n1 = n // 2
    p_base = rng.uniform(0.15, 0.85, m)
    shift = rng.uniform(-0.12, 0.12, m)
    X = np.empty((n, m))
    for k in range(m):
        p1 = np.clip(p_base[k] + shift[k], 0.03, 0.97)
        p2 = np.clip(p_base[k] - shift[k], 0.03, 0.97)
        X[:n1, k] = rng.binomial(2, p1, n1)
        X[n1:, k] = rng.binomial(2, p2, n - n1)
    chrom = np.repeat(np.arange(1, n_chr + 1).astype(str), per_chr)
    pos = np.tile(np.arange(per_chr) * 50000 + 10000, n_chr)
    snp_names = np.array([f"{c}:{p}" for c, p in zip(chrom, pos)])
    Ha, Hd = noia(X)
    af = X.mean(0) / 2

    def zstd(v):
        v = v - v.mean(); s = v.std(); return v / s if s > 1e-9 else v

    z_polyAA = np.zeros(n)
    for _ in range(300):
        i = rng.randint(0, m); j = rng.randint(0, m)
        if chrom[i] != chrom[j]:
            z_polyAA += rng.normal() * Ha[:, i] * Ha[:, j]
    z_polyAA = zstd(z_polyAA)
    add_snps = rng.choice(m, 25, replace=False)
    z_add = zstd((X[:, add_snps] - X[:, add_snps].mean(0)) @ rng.normal(0, 1, 25))

    het = (X == 1).mean(0)
    def high_het_on(c, exclude=()):
        seg = [s for s in range((c - 1) * per_chr, c * per_chr) if s not in exclude]
        return seg[int(np.argmax(het[seg]))]
    used = set()
    i_aa = high_het_on(1); j_aa = high_het_on(4); used |= {i_aa, j_aa}
    i_ad = high_het_on(2, used); j_ad = high_het_on(5, used); used |= {i_ad, j_ad}
    i_dd = high_het_on(3, used); j_dd = high_het_on(6, used); used |= {i_dd, j_dd}
    i_th = high_het_on(1, used); j_th = high_het_on(2, used); used |= {i_th, j_th}
    i_xo = high_het_on(3, used); j_xo = high_het_on(5, used); used |= {i_xo, j_xo}
    truth = {"AA": (i_aa, j_aa), "AD": (i_ad, j_ad), "DD": (i_dd, j_dd),
             "THRESH": (i_th, j_th), "XOR": (i_xo, j_xo)}

    g_aa = zstd(Ha[:, i_aa] * Ha[:, j_aa])
    g_ad = zstd(Ha[:, i_ad] * Hd[:, j_ad])
    g_dd = zstd(Hd[:, i_dd] * Hd[:, j_dd])
    cnt = X[:, i_th] + X[:, j_th]
    g_th = zstd((cnt >= 3).astype(float))
    md_i = X[:, i_xo] if af[i_xo] < 0.5 else 2 - X[:, i_xo]
    md_j = X[:, j_xo] if af[j_xo] < 0.5 else 2 - X[:, j_xo]
    g_xo = zstd((((md_i >= 1).astype(int) + (md_j >= 1).astype(int)) == 1).astype(float))
    z_conc = zstd(g_aa + g_ad + g_dd + g_th + g_xo)

    va, vaa, vc, ve = 0.15, 0.25, 0.20, 0.40
    y = (np.sqrt(va) * z_add + np.sqrt(vaa) * z_polyAA
         + np.sqrt(vc) * z_conc + np.sqrt(ve) * rng.normal(0, 1, n))
    return X, y, chrom, pos, snp_names, truth


def selftest():
    print("#" * 74)
    print("# Self-test on synthetic data: (A) aggregate-kernel prediction gain, "
          "(B) the six pairwise tests, (C) gate, metrics and track consistency")
    print("#" * 74)
    X, y, chrom, pos, snp_names, truth = synth_data(seed=1)
    n, m = X.shape

    cfg = CFG()
    cfg.OUT_DIR = "pig_epik_selftest"
    cfg.BFILE = "___no_such_file___"
    cfg.N_PCS = 3
    cfg.POOL_SIZE = 150
    cfg.MAX_NEAR_PER_ANCHOR = 30
    cfg.CIS_WINDOW_MB = 0.3
    cfg.MAX_CIS_NEIGHBORS = 6
    cfg.KERNEL_TRANS_SAMPLE = 3000
    cfg.XOR_MAX_PAIRS = 3000
    cfg.N_BOOT = 300
    cfg.N_JOBS = 1
    cfg.MIN_CELL_COUNT_EFF = 10
    cfg.HAP_CTRL_TOP_N = 50

    summary, all_score, gate = run_pipeline(X, y, chrom, pos, snp_names, cfg)

    print("\n" + "=" * 74)
    print("[unit test] scoring the injected true pairs directly, to check the "
          "specificity of the six tests")
    print("=" * 74)
    Ha, Hd = noia(X); af = X.mean(0) / 2
    PCs = pca_from_grm(grm_a(X), cfg.N_PCS)
    global_mains = [PCs[:, c] for c in range(PCs.shape[1])]
    ys = (y - y.mean()) / y.std()

    def make_fn(col, typ):
        def f(a, b):
            r = score_pair(a, b, ys, X, Ha, Hd, af, global_mains,
                           min_count=cfg.MIN_CELL_COUNT_EFF,
                           thr_ks=cfg.THRESHOLD_K_RANGE, types_to_compute=[typ])
            return r.get(col, 0.0) or 0.0
        return f

    rng = np.random.RandomState(3)
    checks = []
    for typ, col in [("AA", "F_AA"), ("AD", "F_AD"), ("DD", "F_DD"),
                     ("THRESH", "F_THRESH"), ("XOR", "F_XOR")]:
        fn = make_fn(col, typ)
        rf = []
        for _ in range(300):
            i, j = rng.randint(0, m), rng.randint(0, m)
            if i != j:
                rf.append(fn(min(i, j), max(i, j)))
        rf = np.array([v for v in rf if np.isfinite(v)])
        a, b = min(truth[typ]), max(truth[typ])
        Ft = fn(a, b)
        # The threshold signal is the weakest by construction: count>=k is highly
        # correlated with the two additive doses, so little is left once the
        # additive main effects are conditioned out. It is therefore judged
        # against the q95 of random pairs, the other five against q99.
        qlev = 95 if typ == "THRESH" else 99
        qv = np.percentile(rf, qlev)
        checks.append((f"{typ}: true-pair F exceeds the q{qlev} of random pairs",
                       Ft > qv, f"F={Ft:.1f} vs q{qlev}={qv:.1f}"))

    print("\n" + "=" * 74)
    print("Consistency checks")
    print("=" * 74)
    d = summary["global_epistasis_dPCC_cv"]
    all_checks = [("aggregate kernel K_AA gives a positive CV gain "
                   "(polygenic AA was injected)", d > 0, f"dPCC={d:+.4f}")]
    all_checks += checks

    # the gate produced a complete panel
    all_checks.append(("gate output complete (ten slice kernels)", len(gate) >= 8,
                       f"{len(gate)} slice kernel(s)"))
    if len(gate):
        need_cols = {"dPCC", "CI_low", "q_BH", "stability_cv", "PASS", "classification"}
        all_checks.append(("gate reports FDR, stability and the pre-registered "
                           "classification", need_cols.issubset(set(gate.columns)),
                           "all columns present"
                           if need_cols.issubset(set(gate.columns)) else "columns missing"))
        # classification agrees with the rule: PASS and dPCC > global ->
        # concentrated; PASS and dPCC <= global -> diffuse-compatible
        okc = True
        for _, r in gate.iterrows():
            if r["PASS"] and r["dPCC"] > d and "concentrated" not in r["classification"]:
                okc = False
            if r["PASS"] and r["dPCC"] <= d and "diffuse" not in r["classification"]:
                okc = False
            if (not r["PASS"]) and "negative" not in r["classification"]:
                okc = False
        all_checks.append(("classification follows the pre-registered rule", okc,
                           "classification consistent with the rule"))

    # each track scored only the interaction types assigned to it
    if len(all_score):
        t = all_score.set_index("search_track")
        okt = True; msg = []
        for track, must_zero in [("pool_cross", ["F_DD", "F_THRESH", "F_XOR"]),
                                 ("cis_DD_threshold", ["F_AA", "F_AD", "F_MULT", "F_XOR"]),
                                 ("xor_trans", ["F_AA", "F_AD", "F_DD", "F_MULT", "F_THRESH"])]:
            sub = all_score[all_score["search_track"] == track]
            if len(sub) == 0:
                continue
            for c in must_zero:
                if c in sub.columns and (sub[c].fillna(0).abs() > 1e-9).any():
                    okt = False; msg.append(f"{track}:{c}")
        all_checks.append(("each track scored only its assigned types", okt,
                           "types correct" if okt else f"out of scope: {msg[:3]}"))

    # the genotype-cell filter behaved
    bad = 0
    for col, tcol in [("F_DD", "testable_DD"), ("F_AD", "testable_AD"), ("F_AA", "testable_AA")]:
        if col in all_score.columns and tcol in all_score.columns:
            tt = all_score[tcol].fillna(False).astype(bool)
            bad += int(((all_score[col].fillna(0) > 1e-9) & (~tt)).sum())
    all_checks.append(("cell filter consistent (untestable pairs have F=0)",
                       bad == 0, f"{bad} violating row(s)"))

    # the phantom-epistasis control ran
    has_hap = "F_hapctrl" in all_score.columns and all_score["F_hapctrl"].notna().any()
    all_checks.append(("cis haplotype phantom control was applied", has_hap,
                       f"{int(all_score['F_hapctrl'].notna().sum()) if 'F_hapctrl' in all_score else 0} pair(s) tested"))

    # output files
    need = ["split_assignment.csv", "structure_info.json", "prediction_cv.csv",
            "prediction_cv_delta.csv", "prediction_holdout.csv", "evaluation_metrics.csv",
            "slice_kernel_gating.csv", "aggregate_field_scores.csv",
            "candidate_pool.csv", "epistasis_scores_all.csv", "run_summary.json"]
    miss = [f for f in need if not os.path.exists(os.path.join(cfg.OUT_DIR, f))]
    all_checks.append(("all output files written", len(miss) == 0,
                       "all present" if not miss else f"missing {miss}"))

    # evaluation metrics complete
    ev = os.path.join(cfg.OUT_DIR, "evaluation_metrics.csv")
    if os.path.exists(ev):
        e = pd.read_csv(ev)
        need_m = {"PCC", "R2", "Spearman", "top_gain", "top_overlap", "calib_slope"}
        all_checks.append(("balanced evaluation metrics complete",
                           need_m.issubset(set(e.columns)),
                           ",".join(sorted(need_m & set(e.columns)))))

    print()
    allok = True
    for name, ok, msg in all_checks:
        if not ok:
            allok = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    print("\n" + "=" * 74)
    print("  self-test: " + ("ALL CHECKS PASS" if allok else "FAILURES PRESENT"))
    print("=" * 74)
    return allok


def main():
    ap = argparse.ArgumentParser(
        description="Multi-resolution epistatic kernels with a gate: discovery pipeline")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bfile"); ap.add_argument("--pheno"); ap.add_argument("--trait")
    ap.add_argument("--cov"); ap.add_argument("--qcov")
    ap.add_argument("--cis_mb", type=float); ap.add_argument("--near_kb", type=int)
    ap.add_argument("--pool", type=int); ap.add_argument("--out")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--no_resume", action="store_true",
                    help="ignore the cache and recompute everything")
    ap.add_argument("--no_admixture", action="store_true",
                    help="skip ADMIXTURE and use GRM-PCA + KMeans instead (seconds)")
    ap.add_argument("--admix_threads", type=int)
    ap.add_argument("--admix_kmax", type=int, help="largest K for ADMIXTURE")
    ap.add_argument("--gate_jobs", type=int, help="worker processes for the gate")
    ap.add_argument("--cis_neighbors", type=int,
                    help="maximum neighbours per SNP inside the cis window")
    ap.add_argument("--part1_only", action="store_true",
                    help="stop after Part 1 (fast dPCC screen); no slice kernels, no scan")
    ap.add_argument("--force_scan", action="store_true",
                    help="scan pairs even if no slice kernel passed the gate")
    ap.add_argument("--shared_ckpt", default=None,
                    help="cache directory for trait-independent GRM, structure "
                         "and slice kernels")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    cfg = CFG()
    for src, dst in [(a.bfile, "BFILE"), (a.pheno, "PHENO_FILE"), (a.trait, "TRAIT"),
                     (a.cov, "COV_FILE"), (a.qcov, "QCOV_FILE"), (a.out, "OUT_DIR"),
                     (a.cis_mb, "CIS_WINDOW_MB"), (a.near_kb, "NEAR_WINDOW_KB"),
                     (a.pool, "POOL_SIZE"), (a.jobs, "N_JOBS")]:
        if src is not None:
            setattr(cfg, dst, src)
    if a.no_resume: cfg.RESUME = False
    if a.no_admixture: cfg.USE_ADMIXTURE = False
    if a.admix_threads: cfg.ADMIX_THREADS = a.admix_threads
    if a.admix_kmax: cfg.ADMIX_K_RANGE = range(1, a.admix_kmax + 1)
    if a.gate_jobs: cfg.GATE_N_JOBS = a.gate_jobs
    if a.cis_neighbors: cfg.MAX_CIS_NEIGHBORS = a.cis_neighbors
    if a.part1_only: cfg.PART1_ONLY = True
    if a.force_scan: cfg.RESPECT_GATE = False
    if a.shared_ckpt: cfg.SHARED_CKPT_DIR = a.shared_ckpt
    X, y, chrom, pos, snp_names, extra = load_real(cfg)
    run_pipeline(X, y, chrom, pos, snp_names, cfg, extra_cov=extra)


if __name__ == "__main__":
    main()
