#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mussel mate allocation, step 2
A+D offspring prediction, all-cross allocation, five-generation projection and
an EHVI-UC comparison
================================================================================
The structure-correction decision and the multi-trait G0 of step 1
(mussel_step1.py) are inherited directly, so both steps share one fixed-effect
setting: this script does not choose a number of principal components itself,
it reads structure_mode and structure_n_fixed from the npz that step 1 wrote.
If step 1 reports model_converged = False this script refuses to run, because
an unconverged G0 cannot support an allocation decision.

Identifiers are collapsed under the PLINK FID_IID convention (c1_c1 -> c1),
exactly as in step 1.

If the sensitivity analysis of step 1 shows that the genetic correlation of a
key trait pair is unstable across correction settings, a warning is printed at
the start. The allocation plan is still produced, but that instability has to
be reported in the manuscript.

The Smith-Hazel index uses only diag(G0), the per-trait genetic standard
deviations, and does not depend on the covariances, so it is not directly
affected by unstable rg. The marker effects, however, are back-solved from u of
the multi-trait model and therefore still carry all of G0 indirectly. This
should be stated in the methods.

INPUT   mussel_step1/step2_inputs.npz , ref_all_phased.vcf ,
        breed_all_phased.vcf , cand_sex.csv , trait_imputed.csv
OUTPUT  mussel_step2/
   gebv_breeding.csv              GEBVs of the candidate parents
   index_weights.csv              Smith-Hazel index weights
   all_crosses.csv                every cross, ranked by usefulness
   all_crosses_by_ehvi.csv        every cross, ranked by EHVI
   pareto_front.csv               the Pareto front
   recommended_crosses.csv        recommended plan (usefulness)
   recommended_crosses_ehvi.csv   recommended plan (EHVI)
   multigen_simulation.csv        five-generation projection
"""
import os, time, warnings
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve, eigh, LinAlgError
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

# ==================== Configuration ====================
STEP1_DIR  = "mussel_step1"
STEP1_NPZ  = os.path.join(STEP1_DIR, "step2_inputs.npz")
REF_VCF    = "ref_all_phased.vcf"
BREED_VCF  = "breed_all_phased.vcf"
SEX_FILE   = "cand_sex.csv"
PHENO_FILE = "trait_imputed.csv"

TRAITS = ["phe1","phe2","phe3","phe4","phe5","phe6","phe7","phe14","phe15"]
# phe8 = phe7/phe6 is a ratio and is exactly collinear with phe6 and phe7 on the
# log scale, so it does not enter the multi-trait model. Economic weights: body
# weight, shell weight and the two colour traits get 0.25 each; the redundant
# size traits phe1-5 get 0. They remain in the model and still receive offspring
# predictions, they simply do not enter the selection index, so that the size
# dimension cannot dominate it.
ECON_W = {"phe6":0.25, "phe7":0.25, "phe14":0.25, "phe15":0.25,
          "phe1":0.0, "phe2":0.0, "phe3":0.0, "phe4":0.0, "phe5":0.0}
KEY_TRAITS = ["phe6", "phe7", "phe14", "phe15"]   # BW / sW / dE / %PP

CM_PER_MB  = 1.0
N_GAMETES  = 200
# Selection intensity, derived rather than copied from a livestock default. One
# cross of Hyriopsis cumingii can in principle produce twenty thousand offspring,
# but the number that can actually be reared to phenotyping is N_SCREEN=500. Of
# those, OFF_PER_CROSS enter the next generation, i.e. p = 5/500 = 1%. The
# standard truncation formula i = phi(Phi^-1(1-p))/p gives 2.665 here.
N_SCREEN   = 500
SEL_INT_i  = 2.665
JITTER     = 1e-6

MAX_USE_PER_PARENT = 5
INBREED_CAP        = 0.10
N_RECOMMEND        = 50

N_GEN         = 5
POP_SIZE      = 300
N_CROSS_GEN   = 60
OFF_PER_CROSS = 5
SIM_SNP_MAX   = 6000

# ---- EHVI-UC comparison arm ----
EHVI_N_REF_PTS      = 4000
EHVI_N_SAMPLES      = 200
EHVI_MARGIN         = 0.5
EHVI_SEED           = 20260729
EHVI_N_RECOMMEND    = 50
EHVI_PROGRESS_EVERY = 2000

# ---- safety switches ----
REQUIRE_CONVERGED = True   # refuse to run if the step-1 model did not converge

OUT_DIR = "mussel_step2"
os.makedirs(OUT_DIR, exist_ok=True)


# ==================== I/O, identical to step 1 ====================
def normalize_id(s):
    """Identifier normalisation identical to step 1, including PLINK FID_IID collapsing."""
    s = str(s).strip().lstrip("\ufeff").replace("\r", "").replace("\n", "")
    if s.endswith(".0") and s[:-2].replace("-", "").replace("_", "").isalnum():
        s = s[:-2]
    if "_" in s:
        parts = s.split("_")
        if len(parts) == 2 and parts[0] == parts[1]:
            s = parts[0]
    return s


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


def grm_a(X):
    p = X.mean(0) / 2.0
    Z = X - 2.0 * p
    c = 2.0 * np.sum(p * (1.0 - p)) + 1e-12
    K = (Z @ Z.T) / c
    return K / np.mean(np.diag(K)), Z, c


def build_noia(X):
    pAA = np.mean(X == 0, axis=0); pAa = np.mean(X == 1, axis=0)
    paa = np.mean(X == 2, axis=0)
    mu = pAa + 2 * paa
    Ha = X - mu
    v = pAA * mu**2 + pAa * (1 - mu)**2 + paa * (2 - mu)**2
    mono = v < 1e-12
    H = (X == 1).astype(float)
    b = np.where(mono, 0, (pAa * (1 - mu)) / np.where(mono, 1, v))
    Hd = H - pAa - b * Ha
    Hd[:, mono] = 0.0
    return Ha, Hd, dict(mu_x=mu, mu_H=pAa, beta=b, mono=mono)


def grm_d(Hd):
    K = (Hd @ Hd.T) / (np.sum(np.var(Hd, axis=0)) + 1e-12)
    return K / np.mean(np.diag(K))


def pca_from_grm(K, n):
    if n <= 0:
        return np.zeros((K.shape[0], 0))
    ev, evec = eigh(K)
    o = np.argsort(ev)[::-1][:n]
    return evec[:, o] * np.sqrt(np.maximum(ev[o], 0))


def build_design_from_step1(K_A, n_ind, mode, n_fixed):
    """
    Reuse the structure-correction setting of step 1 exactly.
    No automatic PC selection happens here. Broken-stick selection would pick
    around twenty PCs and contradict the decision step 1 already made.
    """
    if n_fixed <= 0:
        return np.ones((n_ind, 1)), "intercept only (inherited from step 1)"
    return (np.column_stack([np.ones(n_ind), pca_from_grm(K_A, n_fixed)]),
            f"intercept + {n_fixed} PC(s) (inherited from step 1)")


# ==================== Single-trait A+D REML ====================
class REML_AD:
    def __init__(self, jitter=JITTER):
        self.jitter = jitter

    def _neg_loglik(self, theta, y, X_fix, K_list):
        n = len(y)
        try:
            va, vd, ve = np.exp(theta)
            V = (va * K_list[0] + vd * K_list[1] +
                 ve * np.eye(n) + self.jitter * np.eye(n))
            L = cholesky(V, lower=True)
            Vi = cho_solve((L, True), np.eye(n))
            ViX = Vi @ X_fix
            XtViX = X_fix.T @ ViX
            Li = cholesky(XtViX, lower=True)
            P = Vi - ViX @ cho_solve((Li, True), np.eye(X_fix.shape[1])) @ ViX.T
            Py = P @ y
            nll = (2 * np.sum(np.log(np.diag(L))) +
                   2 * np.sum(np.log(np.diag(Li))) + float(y @ Py))
        except LinAlgError:
            return 1e10, np.zeros(3)
        g = np.empty(3)
        for i, K in enumerate(K_list + [None]):
            tr = np.trace(P) if K is None else float(np.einsum('ij,ji->', P, K))
            q = float(Py @ Py) if K is None else float(Py @ (K @ Py))
            g[i] = 0.5 * (tr - q) * np.exp(theta[i])
        return nll, g

    def fit(self, y, X_fix, K_A, K_D, n_restarts=2, maxiter=1000):
        vy = float(np.var(y)); best = None
        rng = np.random.RandomState(0)
        for r in range(n_restarts):
            scale = np.array([0.3, 0.1, 0.6]) if r == 0 else rng.uniform(0.1, 0.5, 3)
            th0 = np.log(np.maximum(vy * scale, 1e-6))
            res = minimize(self._neg_loglik, th0, args=(y, X_fix, [K_A, K_D]),
                           jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-9})
            if best is None or res.fun < best.fun:
                best = res
        self.va, self.vd, self.ve = np.exp(best.x)
        tot = self.va + self.vd + self.ve
        self.h2a = self.va / tot
        self.h2d = self.vd / tot
        self.ll = -0.5 * best.fun
        self._y = y; self._X = X_fix; self._K_A = K_A; self._K_D = K_D
        return self

    def blup_random(self):
        y = self._y; X = self._X; K_A = self._K_A; K_D = self._K_D
        n = len(y)
        V = (self.va * K_A + self.vd * K_D +
             self.ve * np.eye(n) + self.jitter * np.eye(n))
        L = cholesky(V, lower=True)
        Vi = cho_solve((L, True), np.eye(n))
        ViX = Vi @ X
        Li = cholesky(X.T @ ViX, lower=True)
        beta = cho_solve((Li, True), ViX.T @ y)
        Vr = cho_solve((L, True), y - X @ beta)
        return self.va * (K_A @ Vr), self.vd * (K_D @ Vr)


def backsolve(design, K, u, c):
    Kp = np.linalg.pinv(K + 1e-8 * np.eye(K.shape[0]))
    return (design.T @ (Kp @ u)) / c


# ==================== Gamete simulation ====================
def simulate_gametes(H1, H2, chrom, pos, n_gam, cm_per_mb=CM_PER_MB, seed=0):
    n, m = H1.shape
    rng = np.random.RandomState(seed)
    switch = np.empty(m)
    for ch in np.unique(chrom):
        idx = np.where(chrom == ch)[0]
        idx = idx[np.argsort(pos[idx])]
        switch[idx[0]] = 0.5
        d_M = (np.diff(pos[idx]).astype(float) / 1e6) * cm_per_mb / 100.0
        switch[idx[1:]] = 0.5 * (1 - np.exp(-2 * d_M))
    out = np.empty((n, n_gam, m), dtype=np.int8)
    for g in range(n_gam):
        sw = rng.random_sample((n, m)) < switch[None, :]
        which = np.cumsum(sw, axis=1) % 2
        out[:, g, :] = np.where(which == 0, H1, H2)
    return out


def gamete_breeding_values(gam, eff, p_ref):
    centered = gam.astype(np.float32) - p_ref.astype(np.float32)[None, None, :]
    return np.einsum('igm,mt->igt', centered, eff.astype(np.float32), optimize=True)


def smith_hazel(G0, econ_w):
    """Uses diag(G0) only: the index weights do not depend on the genetic covariances and so are not directly affected by unstable antagonism."""
    a = np.array([econ_w[t] for t in TRAITS], dtype=float)
    a = a / a.sum()
    sd = np.sqrt(np.maximum(np.diag(G0), 1e-12))
    return a / sd, a


# ==================== EHVI-UC ====================
def pareto_mask(P):
    n = len(P); keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(P >= P[i], axis=1) & np.any(P > P[i], axis=1)
        dom[i] = False
        if dom.any():
            keep[i] = False
    return keep


def build_reference_cloud(front_points, ref_point, n_pts, seed, margin):
    rng = np.random.default_rng(seed)
    P = front_points[pareto_mask(front_points)]
    box_hi = front_points.max(axis=0) + margin
    pts = rng.uniform(ref_point, box_hi, size=(n_pts, len(ref_point)))
    base_dominated = np.zeros(n_pts, dtype=bool)
    for p in P:
        base_dominated |= np.all(pts <= p, axis=1)
    cell_vol = np.prod(box_hi - ref_point) / n_pts
    return pts, base_dominated, cell_vol


def ehvi_batch(mu_batch, Sigma_batch, ref_pts, base_dominated, cell_vol,
               n_samples, seed, jitter=1e-8, progress_every=2000):
    rng = np.random.default_rng(seed)
    k, d = mu_batch.shape
    free_pts = ref_pts[~base_dominated]
    ehvi = np.empty(k)
    eye = jitter * np.eye(d)
    t0 = time.time()
    for i in range(k):
        try:
            L = np.linalg.cholesky(Sigma_batch[i] + eye)
        except np.linalg.LinAlgError:
            ehvi[i] = 0.0
            continue
        samples = mu_batch[i] + rng.standard_normal((n_samples, d)) @ L.T
        newly = np.all(free_pts[None, :, :] <= samples[:, None, :], axis=2)
        ehvi[i] = newly.sum(axis=1).mean() * cell_vol
        if progress_every and (i + 1) % progress_every == 0:
            el = time.time() - t0
            print(f"      EHVI {i+1}/{k}  {el:.0f}s elapsed  about {el/(i+1)*(k-i-1):.0f}s remaining")
    return ehvi


# ==================== main ====================
def main():
    t0_all = time.time()

    # ---------------- [1/6] load step 1 ----------------
    print("=" * 74)
    print("[1/6] loading the step-1 results and the breeding population")
    print("=" * 74)
    S = np.load(STEP1_NPZ, allow_pickle=True)
    eff_ref   = S["eff"]
    snp_ref   = S["snp"].astype(str)
    chrom_ref = S["chrom"].astype(str)
    pos_ref   = S["pos"].astype(int)
    G0 = S["G0"]; Rg = S["Rg"]; h2 = S["h2"]; Ysd = S["Ysd"]
    print(f"  step-1 model: {str(S['model'])}, additive effect matrix {eff_ref.shape}")

    # --- inherit the structure-correction decision of step 1 ---
    s_mode  = str(S["structure_mode"]) if "structure_mode" in S else "unknown"
    s_nfix  = int(S["structure_n_fixed"]) if "structure_n_fixed" in S else -1
    if s_nfix < 0:
        print("  [warning] the npz carries no structure-correction information.")
        print("         Falling back to intercept only; broken-stick PC selection is never used.")
        s_mode, s_nfix = "intercept_only", 0
    print(f"  inherited structure correction: mode={s_mode}, n_fixed={s_nfix}")

    # --- convergence gate ---
    if "model_converged" in S:
        conv = bool(S["model_converged"])
        print(f"  step-1 model converged: {conv}")
        if not conv and REQUIRE_CONVERGED:
            raise SystemExit(
                "\n[refusing to run] the step-1 model did not converge, so "
                "its G0 cannot support an allocation decision.\n"
                "  Raise EM_TIME_LIMIT or EM_MAX_ITER in step 1 and rerun, or set\n"
                "  REQUIRE_CONVERGED=False to continue in full knowledge of the risk.")

    # --- warn if antagonism is not robust ---
    sens_f = os.path.join(STEP1_DIR, "sensitivity_antagonism.csv")
    if os.path.exists(sens_f):
        sa = pd.read_csv(sens_f, index_col=0)
        unstable = [ix for ix in sa.index
                    if not ((sa.loc[ix] < 0).all() or (sa.loc[ix] > 0).all())]
        weak = [ix for ix in sa.index if sa.loc[ix].abs().max() < 0.10]
        if unstable or weak:
            print("\n  " + "!" * 68)
            print("  [antagonism robustness warning]")
            if unstable:
                print(f"    trait pairs whose sign flips with the correction "
                      f"({len(unstable)}/{len(sa)}): "
                      f"{', '.join(unstable)}")
            if weak:
                print(f"    pairs with |rg| < 0.10 under every setting, i.e. "
                      f"indistinguishable from zero: "
                      f"{', '.join(weak)}")
            print("    -> the allocation plan is still produced, since the index weights use")
            print("       only diag(G0), but any claim about antagonism or its breakability\n"
                  "       must report this instability.")
            print("  " + "!" * 68)

    # --- reference population ---
    rid, rch, rpos, rsnp, RH1, RH2, RX = read_phased_vcf(REF_VCF)
    phe = pd.read_csv(PHENO_FILE, dtype=str)
    phe["IID"] = phe["IID"].map(normalize_id)
    phe = phe.drop_duplicates("IID").set_index("IID")
    for c in TRAITS:
        phe[c] = pd.to_numeric(phe[c], errors='coerce')

    keep = [k for k, i in enumerate(rid) if i in phe.index]
    rid_use = rid[keep]; RX = RX[keep]; RH1 = RH1[keep]; RH2 = RH2[keep]
    Yref = phe.reindex(rid_use)[TRAITS].values.astype(float)
    ok = ~np.isnan(Yref).any(axis=1)
    rid_use = rid_use[ok]; RX = RX[ok]; RH1 = RH1[ok]; RH2 = RH2[ok]; Yref = Yref[ok]
    Yref = (Yref - Yref.mean(0)) / Yref.std(0)
    print(f"  reference population: n={RX.shape[0]}")
    if RX.shape[0] < 50:
        raise SystemExit("[fatal] reference-population identifiers did not match; check the VCF and phenotype IIDs")

    # --- breeding population ---
    bid, bch, bpos, bsnp, BH1, BH2, BX = read_phased_vcf(BREED_VCF)
    sex_df = pd.read_csv(SEX_FILE)
    sex_df["IID"] = sex_df["IID"].map(normalize_id)
    sex = sex_df.drop_duplicates("IID").set_index("IID")["SEX"].astype(str).str.upper()
    n_hit = int(np.sum([i in sex.index for i in bid]))
    if n_hit == 0:
        print("\n[fatal] the breeding VCF sample names do not match the sex file at all.")
        print(f"  first 5 VCF ids: {[repr(x) for x in bid[:5]]}")
        print(f"  first 5 sex-file ids: {[repr(x) for x in list(sex.index)[:5]]}")
        raise SystemExit(1)
    print(f"  breeding population: {n_hit} individual(s) with a recorded sex")

    # --- shared markers ---
    common = np.intersect1d(np.intersect1d(snp_ref, rsnp), bsnp)
    ir = {s: k for k, s in enumerate(snp_ref)}
    rr = {s: k for k, s in enumerate(rsnp)}
    br = {s: k for k, s in enumerate(bsnp)}
    ci = [ir[s] for s in common]; cr = [rr[s] for s in common]; cb = [br[s] for s in common]
    eff = eff_ref[ci]; chrom = chrom_ref[ci]; pos = pos_ref[ci]
    RX = RX[:, cr]; RH1 = RH1[:, cr]; RH2 = RH2[:, cr]
    BX = BX[:, cb]; BH1 = BH1[:, cb]; BH2 = BH2[:, cb]
    print(f"  shared SNPs: {len(common)}")

    has = np.array([i in sex.index for i in bid])
    bid = bid[has]; BX = BX[has]; BH1 = BH1[has]; BH2 = BH2[has]
    sx = sex.reindex(bid).values
    males = np.where(sx == "M")[0]; females = np.where(sx == "F")[0]
    print(f"  {len(males)} sires x {len(females)} dams = {len(males)*len(females)} crosses")
    if len(males) == 0 or len(females) == 0:
        raise SystemExit("[fatal] no sires or no dams")

    # ---------------- [2/6] dominance effects ----------------
    print("\n" + "=" * 74)
    print("[2/6] estimating dominance effects (fixed effects inherited from step 1)")
    print("=" * 74)
    K_A_r, Z_r, c_r = grm_a(RX)
    Xd_r, desc = build_design_from_step1(K_A_r, RX.shape[0], s_mode, s_nfix)
    rank = np.linalg.matrix_rank(Xd_r)
    print(f"  fixed effects: {desc}, shape={Xd_r.shape}, rank={rank}"
          f" ({'full rank' if rank == Xd_r.shape[1] else 'SINGULAR'})")
    assert rank == Xd_r.shape[1], "singular fixed-effect design matrix"

    Ha_r, Hd_r, noia_par = build_noia(RX)
    K_D_r = grm_d(Hd_r)
    c_d = np.sum(np.var(Hd_r, axis=0)) + 1e-12
    dom_eff = np.zeros((len(common), len(TRAITS)))
    n_dom = 0
    for ti, tr in enumerate(TRAITS):
        m1 = REML_AD().fit(Yref[:, ti], Xd_r, K_A_r, K_D_r, maxiter=1000)
        if m1.h2d < 0.01:
            continue
        uA, uD = m1.blup_random()
        dom_eff[:, ti] = backsolve(Hd_r, K_D_r, uD, c_d)
        n_dom += 1
    print(f"  dominance estimated ({n_dom}/{len(TRAITS)} trait(s) with h2d >= 0.01)")

    # ---------------- [3/6] gamete simulation ----------------
    print("\n" + "=" * 74)
    print("[3/6] gamete simulation and parental transmission values")
    print("=" * 74)
    p_ref = RX.mean(0) / 2.0
    t0 = time.time()
    gam = simulate_gametes(BH1, BH2, chrom, pos, N_GAMETES, seed=1)
    gv = gamete_breeding_values(gam, eff, p_ref)
    gmean = gv.mean(axis=1); gvar = gv.var(axis=1)
    del gam
    print(f"  gamete simulation took {time.time()-t0:.0f}s")

    key_idx = [TRAITS.index(k) for k in KEY_TRAITS]
    gv_key = gv[:, :, key_idx]
    gv_key_c = gv_key - gv_key.mean(axis=1, keepdims=True)
    gcov_key = np.einsum('igt,igs->its', gv_key_c, gv_key_c) / (N_GAMETES - 1)
    del gv
    print(f"  gamete covariance in the KEY_TRAITS={KEY_TRAITS} subspace: {gcov_key.shape}")

    gebv = 2.0 * gmean
    pd.DataFrame(gebv, columns=TRAITS, index=bid).to_csv(
        os.path.join(OUT_DIR, "gebv_breeding.csv"))

    # ---------------- [4/6] all crosses and EHVI ----------------
    print("\n" + "=" * 74)
    print("[4/6] offspring prediction for every cross, Smith-Hazel index and EHVI-UC")
    print("=" * 74)
    b_idx, a_norm = smith_hazel(G0, ECON_W)
    pd.DataFrame({"trait": TRAITS, "econ_weight": a_norm, "index_b": b_idx,
                  "genetic_sd": np.sqrt(np.diag(G0))}).to_csv(
        os.path.join(OUT_DIR, "index_weights.csv"), index=False)

    mi, fi = np.meshgrid(males, females, indexing='ij')
    mi = mi.ravel(); fi = fi.ravel(); npair = len(mi)
    add_mean = gmean[mi] + gmean[fi]
    seg_var = gvar[mi] + gvar[fi]

    pm = BX[mi] / 2.0; pf = BX[fi] / 2.0
    het = pm * (1 - pf) + pf * (1 - pm)
    exp_dose = pm + pf
    EHd = (het - noia_par["mu_H"][None, :] -
           noia_par["beta"][None, :] * (exp_dose - noia_par["mu_x"][None, :]))
    EHd[:, noia_par["mono"]] = 0.0
    dom_mean = EHd @ dom_eff
    del pm, pf, het, exp_dose, EHd

    off_mean = add_mean + dom_mean
    off_sd = np.sqrt(np.maximum(seg_var, 0))
    idx_mean = off_mean @ b_idx
    idx_sd = np.sqrt(np.maximum((off_sd**2) @ (b_idx**2), 0))
    usefulness = idx_mean + SEL_INT_i * idx_sd

    K_all, _, _ = grm_a(np.vstack([RX, BX]))
    nb = BX.shape[0]
    K_b = K_all[-nb:, -nb:]
    inbreed = 0.5 * K_b[mi, fi]

    df = pd.DataFrame({"sire": bid[mi], "dam": bid[fi],
                       "index_mean": idx_mean, "index_sd": idx_sd,
                       "usefulness": usefulness, "progeny_inbreeding": inbreed})
    for ti, tt in enumerate(TRAITS):
        df[f"{tt}_mean"] = off_mean[:, ti]
        df[f"{tt}_sd"] = off_sd[:, ti]

    O = off_mean[:, key_idx]
    nd = np.ones(len(O), bool)
    for a in np.argsort(-O.sum(axis=1)):
        if not nd[a]:
            continue
        nd[np.all(O <= O[a], axis=1) & np.any(O < O[a], axis=1)] = False
    df["pareto_optimal"] = nd

    print("\n  [EHVI] building the reference point cloud and computing EHVI ...")
    Sigma_batch = gcov_key[mi] + gcov_key[fi]
    gebv_key = gebv[:, key_idx]
    ref_point = gebv_key.min(axis=0) - EHVI_MARGIN
    ref_pts, base_dom, cell_vol = build_reference_cloud(
        gebv_key, ref_point, EHVI_N_REF_PTS, EHVI_SEED, EHVI_MARGIN)
    print(f"    reference cloud of {EHVI_N_REF_PTS} points, {int(base_dom.sum())} already dominated by the existing parental front")
    t_e = time.time()
    ehvi_vals = ehvi_batch(O, Sigma_batch, ref_pts, base_dom, cell_vol,
                           EHVI_N_SAMPLES, EHVI_SEED,
                           progress_every=EHVI_PROGRESS_EVERY)
    print(f"    EHVI finished in {(time.time()-t_e)/60:.1f} min over {npair} crosses")
    df["ehvi"] = ehvi_vals

    df.sort_values("usefulness", ascending=False).to_csv(
        os.path.join(OUT_DIR, "all_crosses.csv"), index=False)
    df.sort_values("ehvi", ascending=False).to_csv(
        os.path.join(OUT_DIR, "all_crosses_by_ehvi.csv"), index=False)
    df[df["pareto_optimal"]].to_csv(os.path.join(OUT_DIR, "pareto_front.csv"), index=False)

    def pick(score, n_rec):
        use = np.zeros(len(bid), int); chosen = []
        for k in np.argsort(-score):
            if inbreed[k] > INBREED_CAP:
                continue
            s, d = mi[k], fi[k]
            if use[s] >= MAX_USE_PER_PARENT or use[d] >= MAX_USE_PER_PARENT:
                continue
            chosen.append(k); use[s] += 1; use[d] += 1
            if len(chosen) >= n_rec:
                break
        return chosen

    rec = df.iloc[pick(usefulness, N_RECOMMEND)].copy()
    rec.to_csv(os.path.join(OUT_DIR, "recommended_crosses.csv"), index=False)
    rec_e = df.iloc[pick(ehvi_vals, EHVI_N_RECOMMEND)].copy()
    rec_e.to_csv(os.path.join(OUT_DIR, "recommended_crosses_ehvi.csv"), index=False)
    print(f"  {len(rec)} cross(es) recommended by usefulness, {len(rec_e)} by EHVI")
    pk = lambda d_: set(d_["sire"].astype(str) + "_" + d_["dam"].astype(str))
    print(f"  overlap between the two plans: {len(pk(rec) & pk(rec_e))}/{min(len(rec), len(rec_e))}"
          f" (the smaller the overlap, the more EHVI selects crosses usefulness does not see)")

    # ---------------- [5/6] five-generation projection ----------------
    print("\n" + "=" * 74)
    print("[5/6] five-generation projection under usefulness, mean and random")
    print("=" * 74)
    if SIM_SNP_MAX and len(common) > SIM_SNP_MAX:
        rs = np.random.RandomState(7)
        sub = np.sort(rs.choice(len(common), SIM_SNP_MAX, False))
    else:
        sub = np.arange(len(common))
    eff_s = eff[sub]; ch_s = chrom[sub]; ps_s = pos[sub]; p_s = p_ref[sub]

    def one_generation(H1, H2, sx_, strategy, rng):
        gm = simulate_gametes(H1, H2, ch_s, ps_s, 60, seed=rng.randint(10**6))
        gval = gamete_breeding_values(gm, eff_s, p_s)
        gmn = gval.mean(1); gvr = gval.var(1)
        del gm, gval
        M = np.where(sx_ == "M")[0]; F = np.where(sx_ == "F")[0]
        if len(M) == 0 or len(F) == 0:
            return None
        a, bq = np.meshgrid(M, F, indexing='ij')
        a = a.ravel(); bq = bq.ravel()
        im = (gmn[a] + gmn[bq]) @ b_idx
        isd = np.sqrt(np.maximum((gvr[a] + gvr[bq]) @ (b_idx**2), 0))
        if strategy == "usefulness":
            score = im + SEL_INT_i * isd
        elif strategy == "mean":
            score = im
        else:
            score = rng.random(len(im))
        nH1, nH2, nsx = [], [], []
        for k in np.argsort(-score)[:N_CROSS_GEN]:
            s, d = a[k], bq[k]
            # Simulate N_SCREEN offspring of this pair and take the OFF_PER_CROSS
            # with the highest index into the next generation. Drawing them at
            # random would be wrong: the selection intensity i in the usefulness
            # formula has to be realised in the simulation, otherwise the
            # advantage of usefulness over the mean strategy is systematically
            # underestimated.
            gs = simulate_gametes(H1[[s]], H2[[s]], ch_s, ps_s, N_SCREEN,
                                  seed=rng.randint(10**6))[0]
            gd = simulate_gametes(H1[[d]], H2[[d]], ch_s, ps_s, N_SCREEN,
                                  seed=rng.randint(10**6))[0]
            off_dose = (gs.astype(np.float32) + gd.astype(np.float32)
                       - 2*p_s[None, :].astype(np.float32))
            off_idx = (off_dose @ eff_s.astype(np.float32)) @ b_idx.astype(np.float32)
            top = np.argsort(-off_idx)[:OFF_PER_CROSS]
            for o in top:
                nH1.append(gs[o]); nH2.append(gd[o])
                nsx.append("M" if rng.random() < 0.5 else "F")
        nH1 = np.array(nH1, dtype=np.int8); nH2 = np.array(nH2, dtype=np.int8)
        nsx = np.array(nsx)
        if len(nH1) > POP_SIZE:
            sel = rng.choice(len(nH1), POP_SIZE, replace=False)
            nH1 = nH1[sel]; nH2 = nH2[sel]; nsx = nsx[sel]
        return nH1, nH2, nsx

    rows = []
    for strat in ["usefulness", "mean", "random"]:
        rng = np.random.RandomState(2024)
        H1c, H2c, sxc = BH1[:, sub].copy(), BH2[:, sub].copy(), sx.copy()
        for gen in range(N_GEN + 1):
            A = ((H1c + H2c).astype(float) - 2 * p_s[None, :]) @ eff_s
            idxv = A @ b_idx
            row = {"strategy": strat, "generation": gen, "n": len(H1c),
                   "index_mean": float(idxv.mean()), "index_sd": float(idxv.std())}
            for ti, tt in enumerate(TRAITS):
                row[f"{tt}_mean"] = float(A[:, ti].mean())
            rows.append(row)
            if gen == N_GEN:
                break
            out = one_generation(H1c, H2c, sxc, strat, rng)
            if out is None:
                break
            H1c, H2c, sxc = out
    sim = pd.DataFrame(rows)
    sim.to_csv(os.path.join(OUT_DIR, "multigen_simulation.csv"), index=False)

    # ---------------- [6/6] summary ----------------
    print("\n" + "=" * 74)
    print("[6/6] summary")
    print("=" * 74)
    print("mean index per generation:")
    print(sim.pivot_table(index="generation", columns="strategy",
                          values="index_mean").round(3).to_string())
    print("\ntop 5 by usefulness:")
    print(df.sort_values("usefulness", ascending=False)
          [["sire", "dam", "usefulness", "ehvi"]].head(5).to_string(index=False))
    print("\ntop 5 by EHVI:")
    print(df.sort_values("ehvi", ascending=False)
          [["sire", "dam", "usefulness", "ehvi"]].head(5).to_string(index=False))
    print(f"\nstructure correction inherited from step 1: {s_mode} (n_fixed={s_nfix})")
    print(f"total runtime: {(time.time()-t0_all)/60:.1f} min")


if __name__ == "__main__":
    main()
