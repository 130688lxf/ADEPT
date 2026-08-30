#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mussel mate allocation, step 1
Automatic structure decision + multi-trait EM-REML + epistasis discriminator +
breakable-antagonism decomposition + sensitivity analysis
================================================================================
Three things are combined into one script so that the whole step runs without
manual intervention:

  [A] A deterministic decision on population-structure correction
      (decide_structure_correction). The number of PCs is neither chosen by
      hand nor by broken-stick, which treats family relatedness as "significant
      structure" and runs all the way up to MAX_PCS.

  [B] Multi-trait EM-REML (FA(1-3) or unstructured, selected by AIC). G0 is the
      genetic covariance estimated directly by REML, not the sample covariance
      of per-trait marker effects; that proxy systematically underestimates
      genetic correlations, most severely for low-heritability traits.

  [C] Sensitivity analysis: h2 and the genetic correlation of the key
      antagonistic pairs are re-estimated under several correction settings, to
      show whether the conclusions depend on the correction.

--------------------------------------------------------------------------------
FOUR CRITERIA FOR STRUCTURE CORRECTION (details in decide_structure_correction)

  1  Is there non-random structure at all?
     lambda1 against the Marchenko-Pastur null bound (1+sqrt(n/m))^2.

  2  Discrete strata or continuous family structure?
     The spectral gap, i.e. the drop ratio between adjacent eigenvalues. These
     two questions must be asked separately: family relatedness also raises
     eigenvalues, so their magnitude alone cannot distinguish the two.

  3  ADMIXTURE CV, with boundary protection.
     An argmin at either end of the search range is reported as 'boundary' and
     treated as invalid: the minimum was never bracketed, so what was found is
     the artificial edge of the range rather than a real optimum. A total
     relative decline below CV_FLAT_REL is reported as 'flat', meaning no
     discrete ancestral populations. Only an interior minimum with a
     substantial improvement on both sides is accepted as K.

  4  A safety valve. Even with discrete strata confirmed, if the first five PCs
     already explain R2 >= R2_SAFE of some trait, that trait's differences are
     probably real genetic differences between lines (the gold and purple shell
     colour lines differ genuinely), and correcting would delete genetic
     variance as if it were structure. The decision then falls back to 0 PCs
     and is flagged for manual review of the rearing design:
       common rearing    -> line differences are genetic -> do not correct
                            (0 PCs; K_A handles it)
       separate rearing  -> possibly confounded with environment -> use the
                            line label as a fixed factor

  Any contradiction or insufficiency falls back to the safe default of 0 PCs.
  Rationale: in a breeding population K_A IS the genomic relationship matrix
  and family structure is the thing being estimated, not a confounder.
  Removing the leading eigenvectors of K_A as fixed effects assigns the largest
  block of real additive variance to the fixed effects and depresses h2.

--------------------------------------------------------------------------------
INPUT   ref_all_phased.vcf , trait_imputed.csv
OUTPUT  mussel_step1/
   structure_decision.json        the whole decision path, reproducible and
                                  quotable in the methods
   model_selection.csv            AIC, convergence and antagonistic-pair count
                                  of the four models
   model_states.npz               cached G0/R0 per model, so changing the model
                                  choice does not require refitting
   genetic_correlation.csv        genetic correlation matrix
   variance_components.csv        h2 and variance components
   epistasis_discriminator.csv    epistasis discriminator and self-calibrated
                                  detection power
   antagonism_decomposition.csv   breakable-antagonism decomposition with
                                  block-level bootstrap CIs
   sensitivity_h2.csv             h2 under each correction setting
   sensitivity_antagonism.csv     rg of the antagonistic pairs under each setting
   step2_inputs.npz               everything step 2 needs

Dependencies: numpy, scipy, pandas; plink and admixture are optional and
criterion 3 is skipped when they are absent.
"""
import os, json, re, shutil, subprocess, time, warnings
from collections import deque
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve, eigh, LinAlgError
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

# ==================== Configuration ====================
REF_VCF    = "ref_all_phased.vcf"
PHENO_FILE = "trait_imputed.csv"
TRAITS = ["phe1","phe2","phe3","phe4","phe5","phe6","phe7","phe8","phe14","phe15"]
ECON_W = {"phe5":0.2,"phe6":0.2,"phe14":0.2,"phe15":0.2,
          "phe3":0.1,"phe8":0.05,"phe1":0.05,
          "phe2":0.0,"phe4":0.0,"phe7":0.0}
OUT_DIR = "mussel_step1"; os.makedirs(OUT_DIR, exist_ok=True)

FA_ORDERS   = [1, 2, 3]
BLOCK_KB    = 1000
N_BOOT      = 1000
CALIB_LEVELS= [0.00, 0.15]
CALIB_REPS  = 5
JITTER      = 1e-8

# ---- structure-decision parameters; report these values in the methods ----
ADMIX_K_RANGE = range(1, 9)   # candidate K; wide enough that argmin is bracketed
ADMIX_BIN     = "admixture"
PLINK_BIN     = "plink"
CV_FLAT_REL   = 0.05   # total relative decline below this: flat, no discrete strata
CV_MIN_DROP   = 0.01   # minimum relative improvement on each side of an interior minimum
GAP_RATIO     = 3.0    # spectral-gap threshold above which strata are inferred
MP_MULT       = 2.0    # how far lambda1 must exceed the null bound
R2_SAFE       = 0.15   # safety valve: no correction if the first 5 PCs explain more
N_PC_PROBE    = 10

# ---- sensitivity analysis ----
RUN_SENSITIVITY   = True
SENS_PC_SETTINGS  = [0, 5, 10]   # PC counts compared against the decided setting

# ---- EM controls ----
EM_MAX_ITER   = 20000
EM_TOL_LL     = 1e-8
EM_TOL_PAR    = 1e-6
EM_LL_WINDOW  = 50
EM_TIME_LIMIT = 1800
EM_PRINT_EVERY= 200


# ==================== Identifier normalisation ====================
def normalize_id(s):
    """Strip whitespace, BOM and CRLF, drop a trailing .0, and collapse PLINK FID_IID duplication."""
    s = str(s).strip().lstrip("\ufeff").replace("\r", "").replace("\n", "")
    if s.endswith(".0") and s[:-2].replace("-", "").replace("_", "").isalnum():
        s = s[:-2]
    if "_" in s:
        parts = s.split("_")
        if len(parts) == 2 and parts[0] == parts[1]:
            s = parts[0]
    return s


# ==================== Phased VCF reading ====================
def read_phased_vcf(path):
    ids = None; chrom = []; pos = []; snp = []; h1 = []; h2 = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("##"):
                continue
            p = line.split("\t")
            if line.startswith("#CHROM"):
                ids = np.array([normalize_id(x) for x in p[9:]], dtype=str)
                continue
            chrom.append(p[0]); pos.append(int(p[1])); snp.append(p[2])
            a = []; b = []
            for g in p[9:]:
                gt = g.split(":")[0]
                x, y = (gt.split("|") if "|" in gt else gt.split("/"))
                a.append(0 if x == "0" else 1)
                b.append(0 if y == "0" else 1)
            h1.append(a); h2.append(b)
    H1 = np.array(h1, dtype=np.int8).T
    H2 = np.array(h2, dtype=np.int8).T
    X = (H1 + H2).astype(float)
    return ids, np.array(chrom), np.array(pos), np.array(snp), H1, H2, X


# ==================== GRM / NOIA ====================
def build_noia(X):
    pAA = np.mean(X == 0, 0); pAa = np.mean(X == 1, 0); paa = np.mean(X == 2, 0)
    mu = pAa + 2*paa
    Ha = X - mu
    v = pAA*mu**2 + pAa*(1-mu)**2 + paa*(2-mu)**2
    mono = v < 1e-12
    H = (X == 1).astype(float)
    b = np.where(mono, 0, (pAa*(1-mu))/np.where(mono, 1, v))
    Hd = H - pAa - b*Ha
    Hd[:, mono] = 0.
    return Ha, Hd


def grm_a(X):
    p = X.mean(0) / 2
    Z = X - 2*p
    c = 2*np.sum(p*(1-p)) + 1e-12
    K = (Z @ Z.T) / c
    return K / np.mean(np.diag(K)), Z, c


def grm_d(Hd):
    K = (Hd @ Hd.T) / (np.sum(np.var(Hd, 0)) + 1e-12)
    return K / np.mean(np.diag(K))


def hadamard(K1, K2):
    K = K1 * K2
    return K / (np.mean(np.diag(K)) + 1e-12)


def pca_from_grm(K, n):
    if n <= 0:
        return np.zeros((K.shape[0], 0))
    ev, evec = eigh(K)
    o = np.argsort(ev)[::-1][:n]
    return evec[:, o] * np.sqrt(np.maximum(ev[o], 0))


# ==================== [A] ADMIXTURE cross-validation (optional) ====================
def run_admixture_cv(vcf_path, k_range, workdir,
                     plink_bin=PLINK_BIN, admix_bin=ADMIX_BIN):
    """Returns {K: CV_error}, or None if the external tools are missing or fail (criterion 3 is then skipped)."""
    if shutil.which(plink_bin) is None or shutil.which(admix_bin) is None:
        print("  [criterion 3 skipped] plink / admixture executables not found.")
        print("               install: conda install -c bioconda plink admixture")
        return None
    os.makedirs(workdir, exist_ok=True)
    bed = os.path.join(workdir, "ref")
    try:
        subprocess.run([plink_bin, "--vcf", vcf_path, "--make-bed", "--double-id",
                        "--allow-extra-chr", "--out", bed],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # ADMIXTURE assumes near-independent markers, so prune first. The pruned
        # set is used only to estimate Q; REML still uses every SNP.
        subprocess.run([plink_bin, "--bfile", bed, "--indep-pairwise", "50", "5", "0.2",
                        "--allow-extra-chr", "--out", bed + "_pr"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([plink_bin, "--bfile", bed, "--extract", bed + "_pr.prune.in",
                        "--make-bed", "--allow-extra-chr", "--out", bed + "_p"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # ADMIXTURE only accepts integer chromosome codes
        bim = bed + "_p.bim"
        lines = open(bim).read().splitlines()
        cmap = {}; out = []
        for ln in lines:
            fld = ln.split("\t")
            if not fld[0].lstrip("-").isdigit():
                if fld[0] not in cmap:
                    cmap[fld[0]] = len(cmap) + 1
                fld[0] = str(cmap[fld[0]])
            out.append("\t".join(fld))
        open(bim, "w").write("\n".join(out) + "\n")

        cv = {}
        for k in k_range:
            logf = os.path.join(workdir, f"k{k}.log")
            with open(logf, "w") as fo:
                subprocess.run([admix_bin, "--cv", "-s", "1", "ref_p.bed", str(k)],
                               cwd=workdir, check=True, stdout=fo,
                               stderr=subprocess.STDOUT)
            for line in open(logf):
                m = re.search(r"CV error \(K=(\d+)\):\s*([\d.]+)", line)
                if m:
                    cv[int(m.group(1))] = float(m.group(2))
        return cv if cv else None
    except subprocess.CalledProcessError as e:
        print(f"  [criterion 3 skipped] plink/admixture failed: {e}")
        return None


def load_admixture_Q(workdir, best_k, final_ids):
    """
    Read the ADMIXTURE .Q file, align it to final_ids in .fam order and drop
    the last column. Dropping one column is required: the rows of Q sum to 1
    and are therefore perfectly collinear with the intercept, which would make
    the design matrix singular.
    """
    qf = os.path.join(workdir, f"ref_p.{best_k}.Q")
    famf = os.path.join(workdir, "ref_p.fam")
    if not (os.path.exists(qf) and os.path.exists(famf)):
        return None
    Q = np.loadtxt(qf)
    if Q.ndim == 1:
        Q = Q[:, None]
    fam_ids = [normalize_id(l.split()[1]) for l in open(famf)]
    id2row = {v: k for k, v in enumerate(fam_ids)}
    if not all(i in id2row for i in final_ids):
        print("  [warning] .fam does not match the final individual set; Q cannot be aligned, falling back to 0 PCs")
        return None
    Q = Q[[id2row[i] for i in final_ids]]
    return Q[:, :-1]          # one column dropped to break the collinearity


# ==================== [A] Structure-correction decision ====================
def decide_structure_correction(K_A, Y, n_ind, n_snp, cv_curve=None,
                                n_pc_probe=N_PC_PROBE,
                                cv_flat_rel=CV_FLAT_REL, cv_min_drop=CV_MIN_DROP,
                                gap_ratio_thr=GAP_RATIO, mp_mult=MP_MULT,
                                r2_safe=R2_SAFE):
    """
    A deterministic decision. Returns a dict; downstream code acts on
    d['n_fixed'] and d['mode']. A search-boundary value is never returned;
    contradictory or insufficient evidence falls back to the safe default of
    0 PCs.
    """
    out = {}
    ev = np.sort(eigh(K_A, eigvals_only=True))[::-1]
    top = ev[:n_pc_probe]

    # --- criterion 1: is there non-random structure (Marchenko-Pastur bound) ---
    mp_bound = (1.0 + np.sqrt(n_ind / n_snp))**2
    has_structure = bool(top[0] > mp_mult * mp_bound)
    out['eigenvalues_top'] = [float(x) for x in top]
    out['mp_bound'] = float(mp_bound)
    out['lambda1'] = float(top[0])
    out['lambda1_over_null'] = float(top[0] / mp_bound)
    out['has_structure'] = has_structure

    # --- criterion 2: discrete strata or continuous family structure (spectral gap) ---
    ratios = top[:-1] / top[1:]
    out['gap_at'] = int(np.argmax(ratios)) + 1
    out['gap_ratio'] = float(ratios.max())
    discrete_by_gap = bool(out['gap_ratio'] > gap_ratio_thr)
    out['discrete_by_gap'] = discrete_by_gap

    # --- criterion 3: ADMIXTURE CV, with boundary protection ---
    k_admix, cv_status = None, 'absent'
    if cv_curve:
        ks = sorted(cv_curve)
        vals = np.array([cv_curve[k] for k in ks])
        rel_total = float((vals[0] - vals.min()) / abs(vals[0]))
        argmin_k = ks[int(np.argmin(vals))]
        at_boundary = argmin_k in (ks[0], ks[-1])
        out['cv_curve'] = {int(k): float(cv_curve[k]) for k in ks}
        out['cv_rel_total_drop'] = rel_total
        out['cv_argmin_k'] = int(argmin_k)
        out['cv_argmin_at_boundary'] = bool(at_boundary)
        if rel_total < cv_flat_rel:
            cv_status, k_admix = 'flat', 1
        elif at_boundary:
            cv_status, k_admix = 'boundary', None
        else:
            i = int(np.argmin(vals))
            drop_l = (vals[i-1] - vals[i]) / abs(vals[i-1])
            drop_r = (vals[i+1] - vals[i]) / abs(vals[i+1])
            if min(drop_l, drop_r) >= cv_min_drop:
                cv_status, k_admix = 'interior', int(argmin_k)
            else:
                cv_status, k_admix = 'flat', 1
    out['cv_status'] = cv_status
    out['k_admix'] = k_admix

    # --- criterion 4: is correcting safe, or would PCs eat the genetic signal ---
    d_, U_ = eigh(K_A)
    o = np.argsort(d_)[::-1][:n_pc_probe]
    PC = U_[:, o] * np.sqrt(np.maximum(d_[o], 0))
    Yc = (Y - Y.mean(0)) / Y.std(0)
    n = len(Yc)
    r2_by_trait = {}
    for ti in range(Yc.shape[1]):
        Xk = np.column_stack([np.ones(n), PC[:, :5]])
        beta, *_ = np.linalg.lstsq(Xk, Yc[:, ti], rcond=None)
        r2 = 1 - (Yc[:, ti] - Xk @ beta).var() / Yc[:, ti].var()
        r2_by_trait[TRAITS[ti]] = float(r2)
    r2_max = max(r2_by_trait.values())
    r2_arg = max(r2_by_trait, key=r2_by_trait.get)
    out['r2_top5pc_by_trait'] = r2_by_trait
    out['r2_max'] = float(r2_max)
    out['r2_max_trait'] = r2_arg
    safe_to_correct = bool(r2_max < r2_safe)
    out['safe_to_correct'] = safe_to_correct

    # --- combine into a decision ---
    discrete = discrete_by_gap and (cv_status == 'interior')
    if not has_structure:
        out.update(mode='intercept_only', n_fixed=0,
                   reason='eigenvalues do not exceed the null bound; the population is close to unrelated and needs no correction')
    elif discrete_by_gap and cv_status in ('boundary', 'absent'):
        out.update(mode='intercept_only_unconfirmed', n_fixed=0,
                   reason=f"a spectral gap of {out['gap_ratio']:.2f} suggests possible "
                          f"discrete strata, but ADMIXTURE CV={cv_status} (boundary "
                          f"or unavailable) cannot confirm K; falling back to 0 PCs. "
                          f"Widen ADMIX_K_RANGE and rerun before deciding.")
    elif not discrete:
        out.update(mode='intercept_only', n_fixed=0,
                   reason=f"strong structure (lambda1={out['lambda1']:.1f}, "
                          f"{out['lambda1_over_null']:.0f}x the null bound) but of the "
                          f"continuous family type (gap {out['gap_ratio']:.2f}, "
                          f"CV={cv_status}); the random K_A term already absorbs family "
                          f"relatedness and adding PCs would remove real genetic variance")
    elif not safe_to_correct:
        out.update(mode='intercept_only_forced', n_fixed=0,
                   reason=f"discrete strata detected (K={k_admix}), but the first five "
                          f"PCs already explain R2={r2_max:.3f} (>={r2_safe}) of "
                          f"{r2_arg}, so correcting would delete real genetic "
                          f"variance. Forced back to 0 PCs; check the rearing design "
                          f"manually (common rearing implies genetic differences and no "
                          f"correction; separate rearing implies using the line as a "
                          f"fixed factor)")
    else:
        out.update(mode='admixture_Q', n_fixed=int(k_admix - 1),
                   reason=f"discrete strata confirmed (K={k_admix}, gap "
                          f"{out['gap_ratio']:.2f}) and correction is safe (max "
                          f"R2={r2_max:.3f}); using K-1={k_admix-1} columns of Q")
    return out


def build_design(mode, n_fixed, K_A, n_ind, Q=None):
    """Build the fixed-effect design matrix implied by the decision."""
    if n_fixed <= 0:
        return np.ones((n_ind, 1)), "intercept only"
    if mode == 'admixture_Q' and Q is not None and Q.shape[1] == n_fixed:
        return np.column_stack([np.ones(n_ind), Q]), f"intercept + {n_fixed} Q column(s)"
    return (np.column_stack([np.ones(n_ind), pca_from_grm(K_A, n_fixed)]),
            f"intercept + {n_fixed} PC(s)")


# ==================== Parameter counts for AIC ====================
def _n_chol(t):
    return t*(t+1)//2

def _n_fa(t, k):
    return sum(t-j for j in range(k)) + t


# ==================== Matrix helpers ====================
def _make_pd(M, floor=1e-10):
    M = 0.5*(M + M.T)
    try:
        ev, U = np.linalg.eigh(M)
    except np.linalg.LinAlgError:
        return np.eye(M.shape[0])*floor
    return (U * np.maximum(ev, floor)) @ U.T


def _project_fa(G, k, n_iter=60, psi_floor=1e-8, tol=1e-12, psi_init=None):
    """Project a positive-definite G onto the FA(k) manifold: G ~= Lam Lam' + diag(psi)."""
    t = G.shape[0]
    dG = np.maximum(np.diag(G).copy(), psi_floor)
    psi = (np.maximum(psi_init.copy(), psi_floor)
           if (psi_init is not None and psi_init.shape == (t,))
           else np.maximum(dG*0.5, psi_floor))
    Lam = np.zeros((t, k))
    for _ in range(n_iter):
        s = 1.0/np.sqrt(psi)
        A = (G * s[:, None]) * s[None, :]
        A = 0.5*(A + A.T)
        try:
            ev, U = np.linalg.eigh(A)
        except np.linalg.LinAlgError:
            break
        idx = np.argsort(ev)[::-1][:k]
        lam_e = np.maximum(ev[idx] - 1.0, 0.0)
        Lam = (np.sqrt(psi)[:, None]) * U[:, idx] * np.sqrt(lam_e)[None, :]
        psi_new = np.maximum(dG - np.sum(Lam**2, axis=1), psi_floor)
        if np.max(np.abs(psi_new - psi)) < tol:
            psi = psi_new
            break
        psi = psi_new
    return _make_pd(Lam @ Lam.T + np.diag(psi)), Lam, psi


# ==================== [B] EM-REML core ====================
def _reml_pieces(Yr, Xr, d, G0, R0, jitter=JITTER):
    """All quantities REML needs, in the rotated space: y_i ~ N(x_i'B, d_i*G0 + R0)."""
    n, t = Yr.shape
    p = Xr.shape[1]
    I_t = np.eye(t)
    V = d[:, None, None]*G0[None] + R0[None] + jitter*I_t[None]
    jit = jitter
    for _ in range(6):
        try:
            L = np.linalg.cholesky(V)
            break
        except np.linalg.LinAlgError:
            jit *= 100
            V = d[:, None, None]*G0[None] + R0[None] + jit*I_t[None]
    else:
        return None
    logdetV = 2.0*np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)))
    Vinv = np.linalg.inv(V)
    Vinv = 0.5*(Vinv + np.transpose(Vinv, (0, 2, 1)))

    C4 = np.einsum('ia,ib,ijk->ajbk', Xr, Xr, Vinv, optimize=True)
    C = C4.reshape(p*t, p*t)
    C = 0.5*(C + C.T) + jitter*np.eye(p*t)
    try:
        Lc = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return None
    logdetC = 2.0*np.sum(np.log(np.diag(Lc)))

    XtVy = np.einsum('ia,ijk,ik->aj', Xr, Vinv, Yr, optimize=True).reshape(p*t)
    B = cho_solve((Lc, True), XtVy).reshape(p, t)
    Cinv = cho_solve((Lc, True), np.eye(p*t)).reshape(p, t, p, t)

    Rres = Yr - Xr @ B
    W = np.einsum('ijk,ik->ij', Vinv, Rres, optimize=True)
    quad = float(np.sum(Rres * W))
    loglik = -0.5*(logdetV + logdetC + quad)

    M = np.einsum('ia,ib,ajbk->ijk', Xr, Xr, Cinv, optimize=True)
    Pii = Vinv - np.matmul(np.matmul(Vinv, M), Vinv)
    return dict(loglik=loglik, W=W, Pii=Pii, B=B, Vinv=Vinv)


def em_reml(Yr, Xr, d, G0_init, R0_init, fa_k=None,
            max_iter=EM_MAX_ITER, tol_ll=EM_TOL_LL, tol_par=EM_TOL_PAR,
            ll_window=EM_LL_WINDOW, time_limit=EM_TIME_LIMIT,
            print_every=EM_PRINT_EVERY, label="", verbose=True):
    """
    EM-REML. fa_k=None leaves G0 unstructured; fa_k=k projects G0 onto FA(k)
    at every step. Update: G0 <- G0 + (1/q) G0 [sum d_i^2 w_i w_i' - sum d_i P_ii] G0.
            R0 <- R0 + (1/n) R0 [Σ w_i w_i'     - Σ P_ii    ] R0
    Convergence is judged by the spread of the log-likelihood over a sliding
    window, which tolerates the small oscillation the FA projection induces.
    """
    n, t = Yr.shape
    q = max(int(np.sum(d > 1e-10*max(d.max(), 1e-30))), 1)
    G0 = _make_pd(G0_init.copy())
    R0 = _make_pd(R0_init.copy())
    psi_cache = None
    if fa_k is not None:
        G0, _, psi_cache = _project_fa(G0, fa_k)

    ll_hist = deque(maxlen=ll_window)
    t0 = time.time()
    best = None; ll_spread = np.inf; rel_dpar = np.inf
    converged = False; it = 0

    for it in range(1, max_iter+1):
        pieces = _reml_pieces(Yr, Xr, d, G0, R0)
        if pieces is None:
            if verbose:
                print(f"      [{label}] numerical decomposition failed at step {it}; reverting to the last stable solution")
            break
        ll = pieces['loglik']; W = pieces['W']; Pii = pieces['Pii']
        ll_hist.append(ll)
        if best is None or ll > best['loglik']:
            best = dict(loglik=ll, G0=G0.copy(), R0=R0.copy(),
                        B=pieces['B'].copy(), Vinv=pieces['Vinv'], iter=it)

        SG = np.einsum('i,ij,ik->jk', d**2, W, W, optimize=True)
        TG = np.einsum('i,ijk->jk', d, Pii, optimize=True)
        G0_new = G0 + (G0 @ (SG - TG) @ G0) / q
        SR = W.T @ W
        TR = Pii.sum(axis=0)
        R0_new = R0 + (R0 @ (SR - TR) @ R0) / n

        G0_new = _make_pd(G0_new); R0_new = _make_pd(R0_new)
        if fa_k is not None:
            G0_new, _, psi_cache = _project_fa(G0_new, fa_k, psi_init=psi_cache)

        rel_dpar = max(
            np.max(np.abs(G0_new - G0)) / (np.max(np.abs(G0)) + 1e-12),
            np.max(np.abs(R0_new - R0)) / (np.max(np.abs(R0)) + 1e-12))
        ll_spread = ((max(ll_hist) - min(ll_hist)) / (abs(ll) + 1.0)
                     if len(ll_hist) == ll_hist.maxlen else np.inf)
        G0, R0 = G0_new, R0_new

        if verbose and print_every and it % print_every == 0:
            print(f"      [{label}] iter {it:>5}  loglik={ll:>12.4f}  "
                  f"ll_spread={ll_spread:.2e}  rel_dpar={rel_dpar:.2e}")
        if ll_spread < tol_ll and rel_dpar < tol_par:
            converged = True
            if verbose:
                print(f"      [{label}] converged at step {it}  ll_spread={ll_spread:.2e}  "
                      f"rel_dpar={rel_dpar:.2e}")
            break
        if time.time() - t0 > time_limit:
            if verbose:
                print(f"      [{label}] wall-clock limit {time_limit}s reached; stopped at step {it}")
            break

    final_G0 = best['G0'] if best is not None else G0
    final_R0 = best['R0'] if best is not None else R0
    pieces = _reml_pieces(Yr, Xr, d, final_G0, final_R0)
    if pieces is None:
        pieces = dict(loglik=best['loglik'], B=best['B'], Vinv=best['Vinv'])
    return dict(G0=final_G0, R0=final_R0, loglik=pieces['loglik'],
                B=pieces['B'], Vinv=pieces['Vinv'], converged=bool(converged),
                n_iter=it, rel_dll=float(ll_spread), rel_dpar=float(rel_dpar),
                seconds=time.time()-t0)


class MultiTraitREML:
    """Multi-trait REML solved by EM. structure is 'un' or ('fa', k)."""
    def __init__(self, structure=('fa', 2), jitter=JITTER):
        self.structure = structure
        self.jitter = jitter

    def _npar_G(self, t):
        return _n_chol(t) if self.structure == 'un' else _n_fa(t, self.structure[1])

    def fit(self, Y, X, K, eig=None, G0_init=None, R0_init=None,
            label="", verbose=True):
        assert not np.isnan(Y).any(), "the multi-trait model requires complete phenotypes"
        n, t = Y.shape
        if eig is None:
            d, U = eigh(K); d = np.maximum(d, 0.)
        else:
            d, U = eig
        Yr = U.T @ Y; Xr = U.T @ X
        if G0_init is None or R0_init is None:
            S = _make_pd(np.cov(Y, rowvar=False))
            G0_init = 0.35*S if G0_init is None else G0_init
            R0_init = 0.65*S if R0_init is None else R0_init
        fa_k = None if self.structure == 'un' else self.structure[1]
        res = em_reml(Yr, Xr, d, G0_init, R0_init, fa_k=fa_k,
                      label=label, verbose=verbose)
        self.G0_ = res['G0']; self.R0_ = res['R0']
        self.loglik_ = res['loglik']
        self.npar_ = self._npar_G(t) + _n_chol(t)
        self.aic_ = 2*self.npar_ - 2*self.loglik_
        self.bic_ = self.npar_*np.log(n) - 2*self.loglik_
        self.converged_ = res['converged']; self.n_iter_ = res['n_iter']
        self.rel_dll_ = res['rel_dll']; self.rel_dpar_ = res['rel_dpar']
        self.seconds_ = res['seconds']
        self.B_ = res['B']; self._Vinv = res['Vinv']
        self.d_, self.U_ = d, U
        self.Yr_, self.Xr_ = Yr, Xr
        return self

    def genetic_corr(self):
        s = np.sqrt(np.diag(self.G0_))
        return self.G0_ / np.outer(s, s)

    def heritability(self):
        return np.diag(self.G0_) / (np.diag(self.G0_) + np.diag(self.R0_))

    def genetic_values(self):
        Rres = self.Yr_ - self.Xr_ @ self.B_
        ur = np.einsum('jk,i,ikl,il->ij', self.G0_, self.d_, self._Vinv, Rres,
                       optimize=True)
        return self.U_ @ ur


# ==================== Single-trait REML, used by the epistasis discriminator ====================
class REML1:
    def __init__(self, j=1e-6):
        self.j = j

    def _og(self, th, y, X, Ks):
        n = len(y)
        try:
            v = np.exp(th)
            V = v[-1]*np.eye(n) + self.j*np.eye(n)
            for vi, K in zip(v[:-1], Ks):
                V = V + vi*K
            L = cholesky(V, lower=True)
            Vi = cho_solve((L, True), np.eye(n))
            ViX = Vi @ X
            XtViX = X.T @ ViX
            Li = cholesky(XtViX, lower=True)
            P = Vi - ViX @ cho_solve((Li, True), np.eye(X.shape[1])) @ ViX.T
            Py = P @ y
            nll = 0.5*(2*np.sum(np.log(np.diag(L))) +
                       2*np.sum(np.log(np.diag(Li))) + float(y @ Py))
        except LinAlgError:
            return 1e10, np.zeros_like(th)
        g = np.empty(len(th))
        for i, K in enumerate(list(Ks) + [None]):
            if K is None:
                tr = np.trace(P); qq = float(Py @ Py)
            else:
                tr = float(np.einsum('ij,ji->', P, K)); qq = float(Py @ (K @ Py))
            g[i] = 0.5*(tr - qq)*v[i]
        return nll, g

    def fit(self, y, X, Ks, n_restarts=2):
        best = None; vy = float(np.var(y)); nv = len(Ks) + 1
        rng = np.random.RandomState(0)
        for r in range(n_restarts):
            sc = np.ones(nv) if r == 0 else rng.uniform(.3, 2.5, nv)
            th0 = np.log(np.maximum(vy/nv*sc, 1e-6))
            res = minimize(self._og, th0, args=(y, X, Ks), jac=True,
                           method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-9})
            if best is None or res.fun < best.fun:
                best = res
        self.v = np.exp(best.x); self.ll = -best.fun
        return self


def lrt_boundary(ll_full, ll_red, df=1):
    from scipy.stats import chi2
    s = max(2.0*(ll_full - ll_red), 0.0)
    return (0.0, 1.0) if s <= 0 else (s, 0.5*float(chi2.sf(s, df)))


def epistasis_discriminator(Y, Xd, K_A, K_D, K_AA):
    """Per trait: a boundary LRT of A+D against A+D+AA, plus the self-calibrated detection power at this sample size."""
    n = K_A.shape[0]
    LA = np.linalg.cholesky(K_A + 1e-6*np.eye(n))
    LE = np.linalg.cholesky(K_AA + 1e-6*np.eye(n))
    calib = {}
    for lv in CALIB_LEVELS:
        det = []
        for rep in range(CALIB_REPS):
            r = np.random.RandomState(700 + rep*11 + int(lv*100))
            ga = LA @ r.normal(0, 1, n); ga /= ga.std()
            ge = LE @ r.normal(0, 1, n); ge /= ge.std()
            ysim = (np.sqrt(.35)*ga + np.sqrt(lv)*ge +
                    np.sqrt(max(1-.35-lv, .01))*r.normal(0, 1, n))
            red = REML1().fit(ysim, Xd, [K_A, K_D])
            full = REML1().fit(ysim, Xd, [K_A, K_D, K_AA])
            det.append(lrt_boundary(full.ll, red.ll)[1] < 0.05)
        calib[lv] = float(np.mean(det))
    rows = []
    for ti, tr in enumerate(TRAITS):
        y = Y[:, ti]
        red = REML1().fit(y, Xd, [K_A, K_D])
        full = REML1().fit(y, Xd, [K_A, K_D, K_AA])
        v = full.v; tot = v.sum()
        s, p = lrt_boundary(full.ll, red.ll)
        rows.append({"trait": tr, "h2_A": v[0]/tot, "h2_D": v[1]/tot,
                     "h2_AA": v[2]/tot, "LRT": s, "pvalue": p,
                     "include_E": bool(p < 0.05),
                     "calib_FPR_at_0": calib[CALIB_LEVELS[0]],
                     "calib_power_at_%.2f" % CALIB_LEVELS[-1]: calib[CALIB_LEVELS[-1]]})
    return pd.DataFrame(rows)


# ==================== Breakable-antagonism decomposition (block bootstrap) ====================
def make_blocks(chrom, pos, block_kb=BLOCK_KB):
    blk = pos // (block_kb * 1000)
    key = np.unique(np.stack([pd.factorize(chrom)[0], blk]),
                    axis=1, return_inverse=True)[1]
    nblk = int(key.max()) + 1
    order = np.argsort(key, kind='stable')
    starts = np.searchsorted(key[order], np.arange(nblk))
    starts = np.append(starts, len(key))
    members = [order[starts[b]:starts[b+1]] for b in range(nblk)]
    return key, nblk, members


def decompose(eff, key, nblk):
    """total = S'S at block scale, pleio = E'E at SNP scale, linkage = total - pleio."""
    t = eff.shape[1]
    S = np.stack([np.bincount(key, weights=eff[:, j], minlength=nblk)
                  for j in range(t)], axis=1)
    pleio = eff.T @ eff
    total = S.T @ S
    return total, pleio, total - pleio


def breakable_frac(pleio, linkage):
    den = np.abs(pleio) + np.abs(linkage)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 1e-12, np.abs(linkage) / den, np.nan)


def block_bootstrap(eff, members, nblk, n_boot=N_BOOT, seed=0):
    """
    Resample LD blocks with replacement, giving every drawn copy of a block a
    new block identifier. Resampling at SNP level is biased: duplicated SNPs
    create spurious cross terms inside the same LD block, which are counted as
    linkage and widen the CI of breakable_frac until it no longer contains the
    point estimate.
    """
    rng = np.random.RandomState(seed)
    sizes = np.array([len(mb) for mb in members])
    out = np.full((n_boot, eff.shape[1], eff.shape[1]), np.nan)
    for b in range(n_boot):
        pick = rng.randint(0, nblk, nblk)
        idx = np.concatenate([members[p] for p in pick])
        newkey = np.repeat(np.arange(nblk), sizes[pick])
        _, pl, lk = decompose(eff[idx], newkey, nblk)
        out[b] = breakable_frac(pl, lk)
    return out


def snp_effects_from_mt(u, Z, c):
    """Back-solve GBLUP to SNP-BLUP: beta = Z'K^- u / c."""
    K = (Z @ Z.T) / c
    Kp = np.linalg.pinv(K + 1e-8*np.eye(K.shape[0]))
    return (Z.T @ (Kp @ u)) / c


def _is_posdef(M):
    try:
        np.linalg.cholesky(M + 1e-10*np.eye(M.shape[0]))
        return True
    except np.linalg.LinAlgError:
        try:
            return bool(np.all(np.linalg.eigvalsh(M) > -1e-8))
        except np.linalg.LinAlgError:
            return False


# ==================== main ====================
def main():
    t_all = time.time()
    t = len(TRAITS)

    # ---------------- [1/6] read ----------------
    print("="*74)
    print("[1/6] reading the phased VCF and the phenotypes")
    print("="*74)
    ids, chrom, pos, snp, H1, H2, X = read_phased_vcf(REF_VCF)
    print(f"  VCF: {len(ids)} individuals, {len(chrom)} SNPs")

    phe = pd.read_csv(PHENO_FILE, dtype=str)
    phe["IID"] = phe["IID"].map(normalize_id)
    phe = phe.drop_duplicates("IID").set_index("IID")
    for c in TRAITS:
        phe[c] = pd.to_numeric(phe[c], errors="coerce")

    common = sorted(set(ids) & set(phe.index))
    print(f"  identifier match: {len(common)}/{len(ids)}")
    if len(common) < 50:
        print("\n[fatal] identifier matching failed:")
        print(f"  first 5 VCF ids: {[repr(x) for x in ids[:5]]}")
        print(f"  first 5 phenotype ids: {[repr(x) for x in list(phe.index)[:5]]}")
        raise SystemExit(1)

    id2idx = {v: k for k, v in enumerate(ids)}
    keep = np.array([id2idx[i] for i in common])
    ids = ids[keep]; X = X[keep]; H1 = H1[keep]; H2 = H2[keep]
    Y = phe.reindex(common)[TRAITS].values.astype(float)
    ok = ~np.isnan(Y).any(1)
    if (~ok).sum():
        print(f"  {(~ok).sum()} individual(s) with incomplete phenotypes were dropped")
    ids = ids[ok]; X = X[ok]; H1 = H1[ok]; H2 = H2[ok]; Y = Y[ok]
    n, m = X.shape
    print(f"  final sample: n={n}, SNP={m}, traits={t}")
    Ysd = Y.std(0)
    Y = (Y - Y.mean(0)) / Ysd

    # ---------------- [2/6] GRM ----------------
    print("\n" + "="*74)
    print("[2/6] building the GRMs (K_A / K_D / K_AA)")
    print("="*74)
    K_A, Z, c = grm_a(X)
    assert np.all(np.isfinite(K_A)), "K_A contains NaN"
    Ha, Hd = build_noia(X)
    K_D = grm_d(Hd)
    K_AA = hadamard(K_A, K_A)
    for nm, KK in [("K_A", K_A), ("K_D", K_D), ("K_AA", K_AA)]:
        print(f"  {nm} positive definite: {bool(np.all(np.linalg.eigvalsh(KK) > -1e-8))}")
    d_eig, U_eig = eigh(K_A); d_eig = np.maximum(d_eig, 0.)
    eig = (d_eig, U_eig)
    print(f"  K_A eigendecomposition cached (rank={int(np.sum(d_eig > 1e-10*d_eig.max()))})")

    # ---------------- [3/6] structure decision ----------------
    print("\n" + "="*74)
    print("[3/6] population-structure correction: automatic decision")
    print("="*74)
    cv_curve = run_admixture_cv(REF_VCF, ADMIX_K_RANGE,
                                os.path.join(OUT_DIR, "admix"))
    if cv_curve:
        print(f"  ADMIXTURE CV: {{{', '.join(f'{k}:{v:.4f}' for k, v in sorted(cv_curve.items()))}}}")

    dec = decide_structure_correction(K_A, Y, n, m, cv_curve=cv_curve)
    print(f"\n  criterion 1, non-random structure: lambda1={dec['lambda1']:.2f} vs null bound"
          f"{dec['mp_bound']:.3f} ({dec['lambda1_over_null']:.1f}×) -> {dec['has_structure']}")
    print(f"  criterion 2, discrete vs continuous: largest gap={dec['gap_ratio']:.2f} (threshold {GAP_RATIO}) "
          f"-> discrete={dec['discrete_by_gap']}")
    print(f"  criterion 3, ADMIXTURE CV: status={dec['cv_status']}, K={dec['k_admix']}")
    if 'cv_argmin_at_boundary' in dec and dec['cv_argmin_at_boundary']:
        print(f"        [note] argmin K={dec['cv_argmin_k']} sits on the search "
              f"boundary; the minimum was not bracketed and the result is invalid")
    print(f"  criterion 4, safety of correcting: max R2 of the first 5 PCs={dec['r2_max']:.3f} "
          f"({dec['r2_max_trait']}, threshold {R2_SAFE}) -> safe={dec['safe_to_correct']}")
    print(f"\n  >>> decision: mode={dec['mode']}, n_fixed={dec['n_fixed']}")
    print(f"      {dec['reason']}")

    Q = None
    if dec['mode'] == 'admixture_Q':
        Q = load_admixture_Q(os.path.join(OUT_DIR, "admix"), dec['k_admix'], ids)
        if Q is None:
            dec.update(mode='intercept_only_forced', n_fixed=0,
                       reason=dec['reason'] + "; but Q could not be read or aligned, falling back to 0 PCs")
            print(f"  [fallback] {dec['reason']}")

    Xd, desc = build_design(dec['mode'], dec['n_fixed'], K_A, n, Q=Q)
    rank = np.linalg.matrix_rank(Xd)
    print(f"  fixed-effect design: {desc}, shape={Xd.shape}, rank={rank}"
          f" ({'full rank' if rank == Xd.shape[1] else 'SINGULAR'})")
    assert rank == Xd.shape[1], "singular fixed-effect design; check that one column of Q was dropped"

    with open(os.path.join(OUT_DIR, "structure_decision.json"), "w") as f:
        json.dump(dec, f, ensure_ascii=False, indent=2)
    print(f"  full decision path written to {OUT_DIR}/structure_decision.json")

    # ---------------- [4/6] multi-trait REML ----------------
    print("\n" + "="*74)
    print("[4/6] multi-trait EM-REML: FA orders and unstructured, selected by AIC")
    print("="*74)
    fits = {}; rows = []

    print("\n  >>> unstructured")
    mod_un = MultiTraitREML(structure='un').fit(Y, Xd, K_A, eig=eig,
                                                label="un", verbose=True)
    fits["unstructured"] = mod_un
    rows.append(dict(model="unstructured", npar=mod_un.npar_, loglik=mod_un.loglik_,
                     AIC=mod_un.aic_, BIC=mod_un.bic_, converged=mod_un.converged_,
                     G0_posdef=_is_posdef(mod_un.G0_), n_iter=mod_un.n_iter_,
                     rel_dll=mod_un.rel_dll_, rel_dpar=mod_un.rel_dpar_,
                     seconds=round(mod_un.seconds_, 1)))
    print(f"  unstructured: AIC={mod_un.aic_:>9.1f} converged={mod_un.converged_} "
          f"iterations={mod_un.n_iter_} ({mod_un.seconds_:.1f}s)")

    for k in FA_ORDERS:
        name = f"FA({k})"
        print(f"\n  >>> {name}")
        G0_start = _project_fa(mod_un.G0_, k)[0]
        mod = MultiTraitREML(structure=('fa', k)).fit(
            Y, Xd, K_A, eig=eig, G0_init=G0_start, R0_init=mod_un.R0_,
            label=name, verbose=True)
        if not np.all(np.isfinite(mod.G0_)):
            print(f"  [skipped] {name}: G0 contains NaN or Inf")
            continue
        fits[name] = mod
        rows.append(dict(model=name, npar=mod.npar_, loglik=mod.loglik_,
                         AIC=mod.aic_, BIC=mod.bic_, converged=mod.converged_,
                         G0_posdef=_is_posdef(mod.G0_), n_iter=mod.n_iter_,
                         rel_dll=mod.rel_dll_, rel_dpar=mod.rel_dpar_,
                         seconds=round(mod.seconds_, 1)))
        print(f"  {name}: AIC={mod.aic_:>9.1f} converged={mod.converged_} "
              f"iterations={mod.n_iter_} ({mod.seconds_:.1f}s)")

    if not rows:
        raise RuntimeError("every multi-trait model failed to fit; check the data")
    ms = pd.DataFrame(rows).sort_values("AIC")

    neg_counts = {nm: int((md.genetic_corr()[np.triu_indices(t, 1)] < 0).sum())
                  for nm, md in fits.items()}
    ms["n_antagonistic"] = ms["model"].map(neg_counts)
    ms.to_csv(os.path.join(OUT_DIR, "model_selection.csv"), index=False)
    np.savez(os.path.join(OUT_DIR, "model_states.npz"),
             **{f"{nm}__{w}": getattr(md, f"{w}_")
                for nm, md in fits.items() for w in ("G0", "R0")})

    print("\n  model comparison:")
    print(ms[["model", "npar", "loglik", "AIC", "converged", "G0_posdef",
              "n_iter", "n_antagonistic"]].to_string(index=False))
    print("\n  n_antagonistic is the number of trait pairs with a negative genetic correlation.")
    print("    Under FA(1), r_jk is proportional to lambda_j * lambda_k, so loadings of the\n"
          "    same sign force every correlation positive: FA(1) cannot express antagonism.")

    fa_pd = ms[(ms["model"].str.startswith("FA")) & ms["G0_posdef"]]
    fa_ok = fa_pd[fa_pd["converged"]]
    model_converged = True
    if len(fa_ok):
        best_name = fa_ok.iloc[0]["model"]
    elif len(fa_pd):
        # no converged FA model but a positive-definite one exists: take the best
        # by AIC, warn loudly, and let the pipeline continue
        model_converged = False
        best_name = fa_pd.iloc[0]["model"]
        print("\n  [warning] no FA model is both converged and positive definite; "
              "falling back to the best positive-definite model by AIC: "
              f"{best_name}")
        print("         The usual cause is EM_TIME_LIMIT cutting the run short, which is\n"
              "         not the same as failing to converge.")
        print("         Raise EM_TIME_LIMIT or EM_MAX_ITER and rerun. Until then these")
        print("         results validate the workflow only and must not be published.")
    else:
        raise RuntimeError("no positive-definite FA model; check the data and model_selection.csv")
    best = fits[best_name]
    print(f"\n  selected model: {best_name} (converged={model_converged})")
    if neg_counts.get(best_name, 0) == 0:
        print("  [warning] the selected model reports zero antagonistic pairs. Under "
              "FA(1) that is structural, not a finding, and must not be read as an\n"
              "  absence of antagonism.")
    _un = ms[ms['model'] == 'unstructured']
    if len(_un) and _un.iloc[0]['AIC'] < fa_pd.iloc[0]['AIC']:
        print("  (the unstructured model has a lower AIC but degenerates at n around 600;\n"
              "   it is listed for comparison and not selected)")

    Rg = pd.DataFrame(best.genetic_corr(), index=TRAITS, columns=TRAITS)
    Rg.to_csv(os.path.join(OUT_DIR, "genetic_correlation.csv"))
    h2 = best.heritability()
    pd.DataFrame({"trait": TRAITS, "h2": h2, "genetic_var": np.diag(best.G0_),
                  "residual_var": np.diag(best.R0_)}).to_csv(
        os.path.join(OUT_DIR, "variance_components.csv"), index=False)
    print("\n  heritabilities:")
    print(pd.Series(h2, index=TRAITS).round(3).to_string())

    # ---------------- [5/6] epistasis discriminator and antagonism decomposition ----------------
    print("\n" + "="*74)
    print("[5/6] epistasis discriminator and breakable-antagonism decomposition")
    print("="*74)
    ed = epistasis_discriminator(Y, Xd, K_A, K_D, K_AA)
    ed.to_csv(os.path.join(OUT_DIR, "epistasis_discriminator.csv"), index=False)
    print(ed[["trait", "h2_A", "h2_D", "h2_AA", "pvalue", "include_E"]]
          .round(3).to_string(index=False))
    calib_fpr = ed['calib_FPR_at_0'].iloc[0]; calib_power = ed.iloc[0, -1]
    print(f"\n  self-calibration: false-positive rate at zero injection={calib_fpr:.0%}, "
          f"detection rate at injection {CALIB_LEVELS[-1]}={calib_power:.0%}")
    if calib_power < 0.5:
        print("  [important] detection power is low, so include_E=False must not be read")
        print("         as an absence of epistasis. Report it as a power limit at this\n"
              "         sample size.")

    u = best.genetic_values()
    eff = snp_effects_from_mt(u, Z, c)
    key, nblk, members = make_blocks(chrom, pos)
    sizes = np.array([len(mb) for mb in members])
    print(f"\n  LD blocks: {nblk} of {BLOCK_KB} kb, median SNPs per block={int(np.median(sizes))}")
    total, pleio, linkage = decompose(eff, key, nblk)
    point = breakable_frac(pleio, linkage)
    tb = time.time()
    boot = block_bootstrap(eff, members, nblk, n_boot=N_BOOT)
    CI_low = np.nanpercentile(boot, 2.5, axis=0)
    CI_high = np.nanpercentile(boot, 97.5, axis=0)
    print(f"  block bootstrap finished, {N_BOOT} replicates ({time.time()-tb:.1f}s)")

    recs = []
    for i in range(t):
        for j in range(i+1, t):
            pl = float(pleio[i, j]); lk = float(linkage[i, j])
            is_anta = bool(Rg.iloc[i, j] < 0)
            recs.append({
                "trait_A": TRAITS[i], "trait_B": TRAITS[j],
                "genetic_corr": Rg.iloc[i, j], "total_cov": float(total[i, j]),
                "pleiotropic": pl, "linkage": lk,
                "breakable_frac": point[i, j] if is_anta else np.nan,
                "CI_low": CI_low[i, j] if is_anta else np.nan,
                "CI_high": CI_high[i, j] if is_anta else np.nan,
                "CI_contains_point": (bool(CI_low[i, j] <= point[i, j] <= CI_high[i, j])
                                      if is_anta else np.nan),
                "decomp_consistent": bool((pl <= 0 and lk <= 0) or (pl >= 0 and lk >= 0)),
                "antagonistic": is_anta})
    ad = pd.DataFrame(recs)
    out_csv = os.path.join(OUT_DIR, "antagonism_decomposition.csv")
    if os.path.exists(out_csv):
        shutil.copy2(out_csv, out_csv.replace(".csv", "_PREV.csv"))
    ad.to_csv(out_csv, index=False)
    anta = ad[ad["antagonistic"]].sort_values("breakable_frac", ascending=False)
    print(f"\n  antagonistic trait pairs: {len(anta)}/{len(ad)}")
    if len(anta):
        print(anta[["trait_A", "trait_B", "genetic_corr", "breakable_frac",
                    "CI_low", "CI_high", "CI_contains_point"]].round(3).to_string(index=False))
        n_ok = int(anta["CI_contains_point"].sum())
        print(f"  check: the point estimate lies inside the CI for {n_ok}/{len(anta)} pair(s)"
              + ("  as expected" if n_ok == len(anta) else "  UNEXPECTED, investigate"))

    # ---------------- [6/6] sensitivity analysis ----------------
    sens_h2 = sens_rg = None
    if RUN_SENSITIVITY:
        print("\n" + "="*74)
        print("[6/6] sensitivity analysis: do the conclusions depend on the correction")
        print("="*74)
        settings = [("decided setting (%s, n_fixed=%d)" % (dec['mode'], dec['n_fixed']), Xd)]
        for npc in SENS_PC_SETTINGS:
            if npc == dec['n_fixed'] and dec['mode'] != 'admixture_Q':
                continue
            Xs, ds = build_design('pc', npc, K_A, n)
            settings.append((f"{ds} ({npc} PC)", Xs))
        print(f"  re-estimating {best_name} under {len(settings)} setting(s); each "
              f"costs about as much as the main model")

        anta_pairs = [(r.trait_A, r.trait_B) for r in anta.itertuples()]
        h2_tab = {}; rg_tab = {}
        for lab, Xs in settings:
            print(f"\n  >>> setting: {lab}")
            fa_k = None if best_name == "unstructured" else int(best_name[3])
            structure = 'un' if fa_k is None else ('fa', fa_k)
            md = MultiTraitREML(structure=structure).fit(
                Y, Xs, K_A, eig=eig, label=lab, verbose=False)
            h2_tab[lab] = md.heritability()
            R = pd.DataFrame(md.genetic_corr(), index=TRAITS, columns=TRAITS)
            rg_tab[lab] = [R.loc[a, b] for a, b in anta_pairs]
            print(f"      converged={md.converged_}, iterations={md.n_iter_}, {md.seconds_:.0f}s")

        sens_h2 = pd.DataFrame(h2_tab, index=TRAITS)
        sens_h2.to_csv(os.path.join(OUT_DIR, "sensitivity_h2.csv"))
        print("\n  h2 under each setting:")
        print(sens_h2.round(3).to_string())
        rng_ = sens_h2.max(axis=1) - sens_h2.min(axis=1)
        print(f"\n  largest range of h2 across settings: {rng_.max():.3f} ({rng_.idxmax()})")
        if rng_.max() < 0.05:
            print("  h2 is insensitive to the correction for every trait; the conclusion is robust")
        else:
            print("  some traits are sensitive to the correction and this must be discussed")

        if anta_pairs:
            sens_rg = pd.DataFrame(rg_tab,
                                   index=[f"{a}-{b}" for a, b in anta_pairs])
            sens_rg.to_csv(os.path.join(OUT_DIR, "sensitivity_antagonism.csv"))
            print("\n  genetic correlation of the antagonistic pairs under each setting:")
            print(sens_rg.round(3).to_string())
            still_neg = (sens_rg < 0).all(axis=1)
            print(f"\n  pairs still negative under every setting: {int(still_neg.sum())}/{len(sens_rg)}")
            if still_neg.all():
                print("  the antagonism result is robust to the correction")
            else:
                print("  some pairs change sign with the correction and this must be reported")
                print("    (low-heritability traits are the most exposed to structure-induced\n"
                      "     spurious correlations)")

    # ---------------- pack for step 2 ----------------
    np.savez(os.path.join(OUT_DIR, "step2_inputs.npz"),
             model=np.array(best_name), traits=np.array(TRAITS),
             G0=best.G0_, R0=best.R0_, Rg=Rg.values, h2=h2,
             ids=ids, u=u, eff=eff, Ysd=Ysd,
             chrom=chrom, pos=pos, snp=snp,
             structure_mode=np.array(dec['mode']),
             structure_n_fixed=np.array(dec['n_fixed']),
             model_converged=np.array(model_converged))

    print("\n" + "="*74)
    print(f"finished; results are in {OUT_DIR}/")
    print(f"  structure correction: {dec['mode']} (n_fixed={dec['n_fixed']})")
    print(f"  model: {best_name} (converged={model_converged})   antagonistic pairs: {len(anta)}/{len(ad)}")
    if not model_converged:
        print("  [warning] the model did not converge; raise EM_TIME_LIMIT and rerun before using these results")
    print(f"  total runtime {(time.time()-t_all)/60:.1f} min")
    print("="*74)


if __name__ == "__main__":
    main()
