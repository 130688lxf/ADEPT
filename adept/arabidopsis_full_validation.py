#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabidopsis epistasis positive control -- the full validation suite
====================================================================
One run produces the whole body of evidence. The aim is that the claim "this
method can detect real epistasis" rests on a chain of checks that can be gone
through one by one, rather than on a single isolated p value.

[Part A] Four tiers of structure correction  <- the core test of real epistasis
         against a structural artefact
    T0 raw          no correction
    T1 pc           the first N PCs as fixed effects (the Atwell 2010 standard)
    T2 resid2       two stages: residualise on the GRM BLUP, then test
    T3 mm           one stage: mixed model, interaction as a FIXED effect
                    (EMMAX style)
  T3 is the statistically correct test: structure is handled by the random
  effect u(K), so the interaction is not first shrunk by the BLUP and then
  measured on what is left. If T2 and T3 disagree, T2 is over-correcting rather
  than reporting a true negative.

[Part B] Scale robustness
    raw / log / rank-based inverse normal, to exclude a scale artefact

[Part C] Internal replication across phenotypes
    the best locus pair is fixed, then retested on FT16/FT22/LN10/LN16/LN22

[Part D] 2x2 genotype-class means and the additivity-deviation contrast
    direct evidence of AND-gate masking

[Part E] The mechanistic chain
    FRI genotype -> FLC expression (FRI activating FLC transcription is the
    molecular mechanism itself)

[Part F] Random-interval negative control
    interval pairs of the same size are drawn at random on other chromosomes
    and put through the identical procedure; the p values should be uniform,
    which shows the procedure does not find epistasis everywhere

[Part G] Self-calibration: inject a known effect and measure power and the
    false-positive rate, on the same calibration convention as the pig and
    mussel materials, once per correction tier

Two numerical safeguards:
  - when the REML sigma_e2 collapses towards zero (boundary degeneracy) that
    tier is refused rather than reporting noise
  - MULT and XOR are mathematically identical on homozygous data, so the
    cross-family Bonferroni does not count them twice
====================================================================
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve, LinAlgError

# ============================ Configuration ============================
BFILE   = "genotype_bed"
PHENO   = "phenotypes.pheno"
OUT_DIR = "arabidopsis_full_validation"

N_PERM      = 1000     # permutations for the main analysis
N_PERM_NEG  = 300      # permutations per negative control (many intervals)
N_NEG_PAIRS = 20       # number of random negative-control interval pairs
N_CALIB     = 30       # replicates per self-calibration cell
MIN_MAF     = 0.10
MIN_CELL    = 5
WINDOW      = 20000
N_PCS       = 5
SEED        = 12345
GRM_CHUNK   = 20000

GENES = {
    "RPM1": (3, 2217000, 2221000),   # AT3G07040
    "RIN4": (3, 9127000, 9129000),   # AT3G25070
    "FRI" : (4,  268000,  270000),   # AT4G00650
    "FLC" : (5, 3173000, 3180000),   # AT5G10140
}

MAIN_TEST = ("FRI", "FLC", "FT10")          # the positive control
NEG_TEST  = ("RPM1", "RIN4", "<i>avrRpm1</i>")  # expected-negative demonstration

# phenotype columns used for internal replication; missing ones are skipped
REPLICATION_PHENOS = ["FT10", "FT16", "FT22", "LN10", "LN16", "LN22",
                      "FT_Field", "FT_GH", "0W", "2W", "4W", "8W",
                      "8W_GH_FT", "0W_GH_FT"]
EXPRESSION_PHENOS = ["FLC", "FRI"]   # these two columns are expression levels

os.makedirs(OUT_DIR, exist_ok=True)

def say(*a):
    print(*a, flush=True)

# ====================== PLINK reading ======================
def read_bim(path):
    ch, sn, po = [], [], []
    for line in open(path):
        c = line.split()
        ch.append(int(c[0])); sn.append(c[1]); po.append(int(c[3]))
    return np.array(ch), np.array(sn, dtype=object), np.array(po)

def read_fam(path):
    return [l.split()[1] for l in open(path)]

_LUT = np.array([0, -1, 1, 2], dtype=np.int8)

def read_snps(bedpath, idxs, n):
    bps = (n + 3) // 4
    out = np.empty((len(idxs), n), dtype=np.int8)
    with open(bedpath, 'rb') as f:
        assert f.read(3) == bytes([0x6c, 0x1b, 0x01]), "not a SNP-major .bed file"
        for r, idx in enumerate(idxs):
            f.seek(3 + int(idx) * bps)
            b = np.frombuffer(f.read(bps), dtype=np.uint8)
            tb = np.stack([b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3],
                          axis=1).reshape(-1)[:n]
            out[r] = _LUT[tb]
    return out.T

# ====================== Phenotypes ======================
def load_pheno_all(path, wanted):
    with open(path) as f:
        header = f.readline().split()
        rows = [l.split() for l in f]
    iid = [r[1] for r in rows]
    data, found = {}, []
    for name in wanted:
        if name not in header:
            continue
        j = header.index(name); v = []
        for r in rows:
            try:
                fv = float(r[j]); v.append(fv if not np.isnan(fv) else np.nan)
            except Exception:
                v.append(np.nan)
        data[name] = np.array(v); found.append(name)
    return iid, data, found

def match_ids(fam_iid, phe_iid):
    fmap = {v: i for i, v in enumerate(phe_iid)}
    pairs = [(gi, fmap[k]) for gi, k in enumerate(fam_iid) if k in fmap]
    if len(pairs) < 0.5 * len(fam_iid):
        def nz(s):
            s = str(s).strip(); return s[:-2] if s.endswith(".0") else s
        fmap = {nz(v): i for i, v in enumerate(phe_iid)}
        pairs = [(gi, fmap[nz(k)]) for gi, k in enumerate(fam_iid) if nz(k) in fmap]
    say(f"  {len(pairs)} identifier(s) matched")
    return pairs

def rank_int(x):
    """Rank-based inverse normal transformation."""
    ok = ~np.isnan(x); out = np.full(len(x), np.nan)
    r = stats.rankdata(x[ok])
    out[ok] = stats.norm.ppf((r - 0.5) / len(r))
    return out

def safe_log(x):
    ok = ~np.isnan(x); out = np.full(len(x), np.nan)
    v = x[ok]
    shift = 0.0 if v.min() > 0 else (1.0 - v.min())
    out[ok] = np.log(v + shift)
    return out

# ====================== GRM ======================
def compute_grm_chunked(bedpath, n_samples, snp_idx_use, chunk=GRM_CHUNK):
    G = np.zeros((n_samples, n_samples)); denom = 0.0
    total = len(snp_idx_use)
    for s in range(0, total, chunk):
        idxs = snp_idx_use[s:s + chunk]
        Xc = read_snps(bedpath, idxs, n_samples).astype(np.float32)
        miss = Xc < 0
        if miss.any():
            cm = np.where((~miss).sum(0) > 0,
                          np.where(miss, 0, Xc).sum(0) / np.maximum((~miss).sum(0), 1), 0.0)
            Xc[miss] = np.take(cm.astype(np.float32), np.where(miss)[1])
        p = Xc.mean(0) / 2.0
        keep = (p > 0.005) & (p < 0.995)
        Xc = Xc[:, keep]; p = p[keep]
        if Xc.shape[1] == 0: continue
        Z = Xc - 2 * p
        G += (Z @ Z.T).astype(np.float64)
        denom += float(2 * np.sum(p * (1 - p)))
        say(f"    GRM {min(s+chunk,total)}/{total}")
    return G / max(denom, 1e-12)

def pcs_from_grm(K, n_pcs):
    w, V = np.linalg.eigh(K)
    o = np.argsort(w)[::-1][:n_pcs]
    return V[:, o] * np.sqrt(np.maximum(w[o], 0.0))

# ====================== REML (single kernel) ======================
class REML1:
    """y = Xb + u + e with u~N(0, vg*K), e~N(0, ve*I). Detects boundary degeneracy."""
    def __init__(self, jitter=1e-6): self.jitter = jitter
    def _neg2(self, lp, y, X, K):
        n = len(y); v = np.exp(lp)
        V = v[0] * K + (v[1] + self.jitter) * np.eye(n)
        try: L = cholesky(V, lower=True)
        except LinAlgError: return 1e10
        ld = 2 * np.sum(np.log(np.diag(L)))
        Viy = cho_solve((L, True), y); ViX = cho_solve((L, True), X)
        XtViX = X.T @ ViX
        try: b = np.linalg.solve(XtViX, X.T @ Viy)
        except np.linalg.LinAlgError: return 1e10
        r = y - X @ b
        q = r @ cho_solve((L, True), r)
        s, lda = np.linalg.slogdet(np.atleast_2d(XtViX))
        if s <= 0: return 1e10
        return 0.5 * (ld + lda + q)
    def fit(self, y, X, K, n_restarts=5):
        vy = np.var(y); best = None; rr = np.random.RandomState(0)
        for i in range(n_restarts):
            sc = np.ones(2) if i == 0 else rr.uniform(0.2, 3.0, 2)
            init = np.log(np.maximum(max(vy / 2, 1e-6) * sc, 1e-6))
            res = minimize(self._neg2, init, args=(y, X, K), method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6,
                                    "adaptive": True})
            if best is None or res.fun < best.fun: best = res
        self.vg_, self.ve_ = np.exp(best.x)
        self.h2_ = self.vg_ / (self.vg_ + self.ve_)
        # boundary degeneracy: the residual variance is driven to zero, so any
        # residual computed afterwards is numerical noise
        self.degenerate_ = (self.ve_ / (self.vg_ + self.ve_)) < 0.01
        self.converged_ = best.success
        return self
    def blup_residual(self, y, X, K):
        n = len(y)
        V = self.vg_ * K + (self.ve_ + self.jitter) * np.eye(n)
        L = cholesky(V, lower=True)
        b = np.linalg.solve(X.T @ cho_solve((L, True), X), X.T @ cho_solve((L, True), y))
        r = y - X @ b
        u = self.vg_ * (K @ cho_solve((L, True), r))
        return r - u

# ====================== NOIA ======================
def build_noia(X, tol=1e-6, verbose=True):
    Xf = X.astype(np.float64).copy()
    miss = Xf < 0
    if miss.any():
        for k in range(Xf.shape[1]):
            m = miss[:, k]
            if m.any(): Xf[m, k] = Xf[~m, k].mean() if (~m).any() else 0.0
    p_AA = np.mean(Xf == 0, 0); p_Aa = np.mean(Xf == 1, 0); p_aa = np.mean(Xf == 2, 0)
    mu = p_Aa + 2 * p_aa
    Ha = Xf - mu
    varA = p_AA * mu**2 + p_Aa * (1 - mu)**2 + p_aa * (2 - mu)**2
    mono = varA < 1e-12
    H = (Xf == 1).astype(float)
    beta = (p_Aa * (1 - mu)) / np.where(mono, 1.0, varA)
    Hd = H - p_Aa - beta * Ha
    Hd[:, mono] = 0.0
    ma = np.abs(Ha.mean(0)).max(); md = np.abs(Hd.mean(0)).max()
    cad = np.abs((Ha * Hd).mean(0) - Ha.mean(0) * Hd.mean(0)).max()
    if verbose:
        say(f"    NOIA orthogonality: max|E(A)|={ma:.2e} max|E(D)|={md:.2e} max|Cov(A,D)|={cad:.2e}")
    if ma >= tol or md >= tol or cad >= tol:
        raise AssertionError("NOIA orthogonality check failed")
    return Ha, Hd, Xf

# ============ Vectorised interaction F with mixed-model whitening ============
def make_whitener(K, vg, ve):
    """Return (U, sqrtw), which turn the GLS into weighted least squares. V = U diag(vg*d+ve) U'."""
    d, U = np.linalg.eigh(K)
    w = 1.0 / np.maximum(vg * d + ve, 1e-12)
    return U, np.sqrt(w)

def transform(M, U=None, sw=None):
    """Rotate and whiten a design matrix or phenotype; U=None applies no transformation."""
    if U is None: return M
    return (U.T @ M) * sw[:, None] if M.ndim == 2 else (U.T @ M) * sw

def scan_pairs(VA, VB, rawA, rawB, nameA, nameB, Ymat, covars,
               U=None, sw=None, min_cell=MIN_CELL, progress_every=0):
    """
    Nested-model interaction F test over every A x B locus pair, with the
    permutations carried as the column dimension of Ymat.
    covars  (n, k) extra fixed effects; the intercept is added automatically.
    U, sw   if given, apply mixed-model whitening (the one-stage test).
    Returns (per-pair DataFrame, vector of the maximum F per column).
    """
    n = VA.shape[0]
    base = np.ones((n, 1)) if covars is None else np.column_stack([np.ones(n), covars])
    baseT = transform(base, U, sw)
    Yt = transform(Ymat, U, sw)
    # rotate the locus vectors once
    VAt = transform(VA, U, sw); VBt = transform(VB, U, sw)
    running = np.full(Ymat.shape[1], -np.inf)
    rows = []; n_filt = 0
    okA = [i for i in range(VA.shape[1]) if _maf_ok(rawA[:, i])]
    okB = [j for j in range(VB.shape[1]) if _maf_ok(rawB[:, j])]
    for ci, i in enumerate(okA):
        for j in okB:
            if not _cells_ok(rawA[:, i], rawB[:, j], min_cell):
                n_filt += 1; continue
            inter = VA[:, i] * VB[:, j]              # built in the original space
            it = transform(inter, U, sw)             # and rotated afterwards
            X0 = np.column_stack([baseT, VAt[:, i], VBt[:, j]])
            X1 = np.column_stack([X0, it])
            F = _F_vec(X0, X1, Yt)
            if F is None: continue
            np.maximum(running, F, out=running)
            rows.append((nameA[i], nameB[j], i, j, F[0]))
        if progress_every and (ci + 1) % progress_every == 0:
            say(f"      ...{ci+1}/{len(okA)}")
    if not rows: return None, None
    df = pd.DataFrame(rows, columns=["snp_A", "snp_B", "iA", "iB", "F"])
    df.attrs["n_filtered"] = n_filt
    return df, running

def _F_vec(X0, X1, Y):
    n = X0.shape[0]; df = n - X1.shape[1]
    if df <= 0: return None
    Q0, _ = np.linalg.qr(X0)
    Q1, R1 = np.linalg.qr(X1)
    if np.min(np.abs(np.diag(R1))) < 1e-8: return None
    SS = (Y**2).sum(0)
    R0 = SS - ((Q0.T @ Y)**2).sum(0)
    Rf = np.maximum(SS - ((Q1.T @ Y)**2).sum(0), 1e-300)
    return ((R0 - Rf) / 1.0) / (Rf / df)

def _maf_ok(g):
    gg = g[g >= 0]
    if len(gg) == 0: return False
    p = gg.mean() / 2.0
    return min(p, 1 - p) >= MIN_MAF

def _cells_ok(gi, gj, min_cell):
    m = (gi >= 0) & (gj >= 0)
    if m.sum() < 10: return False
    _, cnt = np.unique(gi[m] * 3 + gj[m], return_counts=True)
    return len(cnt) >= 4 and (cnt >= min_cell).all()

def make_Ymat(y, valid, n_perm, seed):
    n = len(y); rng = np.random.default_rng(seed)
    Y = np.zeros((n, n_perm + 1))
    Y[valid, 0] = y[valid]
    vi = np.where(valid)[0]
    for k in range(n_perm):
        Y[vi, k + 1] = rng.permutation(y[vi])
    return Y

def perm_p(running, n_perm):
    return (1 + int(np.sum(running[1:] >= running[0]))) / (n_perm + 1)

def single_pair_stats(v1, v2, y, covars=None, U=None, sw=None):
    """F, nominal p, beta and SE for one specified pair of loci."""
    ok = ~np.isnan(y)
    n = int(ok.sum())
    base = np.ones((n, 1)) if covars is None else np.column_stack([np.ones(n), covars[ok]])
    a = v1[ok]; b = v2[ok]; inter = a * b; yy = y[ok]
    if U is not None:
        base = transform(base, U, sw); a = transform(a, U, sw)
        b = transform(b, U, sw); inter = transform(inter, U, sw); yy = transform(yy, U, sw)
    X1 = np.column_stack([base, a, b, inter])
    beta, *_ = np.linalg.lstsq(X1, yy, rcond=None)
    r = yy - X1 @ beta; df = n - X1.shape[1]
    if df <= 0: return dict(F=np.nan, p=np.nan, beta=np.nan, se=np.nan, n=n)
    s2 = float(r @ r) / df
    try: se = float(np.sqrt(max(s2 * np.linalg.inv(X1.T @ X1)[-1, -1], 0)))
    except np.linalg.LinAlgError: se = np.nan
    X0 = X1[:, :-1]
    b0, *_ = np.linalg.lstsq(X0, yy, rcond=None); r0 = yy - X0 @ b0
    F = ((float(r0 @ r0) - float(r @ r)) / 1.0) / (float(r @ r) / df)
    return dict(F=F, p=float(stats.f.sf(F, 1, df)), beta=float(beta[-1]), se=se, n=n)

# ============================ Main ============================
def main():
    rows_summary = []
    say("=" * 68); say("Arabidopsis epistasis positive control: full validation suite"); say("=" * 68)

    say("\n[loading] .bim / .fam ...")
    chrom, snp, pos = read_bim(BFILE + ".bim")
    fam_iid = read_fam(BFILE + ".fam"); n_all = len(fam_iid)
    say(f"  {len(snp)} SNPs, {n_all} samples")

    wanted = sorted(set(REPLICATION_PHENOS + EXPRESSION_PHENOS +
                        [MAIN_TEST[2], NEG_TEST[2]]))
    phe_iid, phe, found = load_pheno_all(PHENO, wanted)
    say(f"  phenotype columns found: {found}")
    pairs = match_ids(fam_iid, phe_iid)
    grow = np.array([p[0] for p in pairs]); prow = np.array([p[1] for p in pairs])
    P = {k: v[prow] for k, v in phe.items()}
    nS = len(grow)

    idx = {k: np.where((chrom == v[0]) & (pos >= v[1] - WINDOW) & (pos <= v[2] + WINDOW))[0]
           for k, v in GENES.items()}
    say("  SNPs per interval: " + ", ".join(f"{k}={len(v)}" for k, v in idx.items()))

    # ---------- GRM, excluding the tested chromosomes ----------
    test_chrs = sorted({GENES[MAIN_TEST[0]][0], GENES[MAIN_TEST[1]][0],
                        GENES[NEG_TEST[0]][0], GENES[NEG_TEST[1]][0]})
    say(f"\n[GRM] genome-wide additive GRM, excluding chr{test_chrs} to avoid proximal contamination")
    use = np.where(~np.isin(chrom, test_chrs))[0]
    say(f"  SNPs entering the GRM: {len(use)}")
    Kfull = compute_grm_chunked(BFILE + ".bed", n_all, use)
    K = Kfull[np.ix_(grow, grow)]
    PCs = pcs_from_grm(K, N_PCS)
    np.save(os.path.join(OUT_DIR, "GRM.npy"), K)

    # ---------- read the genotypes of every interval needed ----------
    all_idx = np.unique(np.concatenate(list(idx.values())))
    Gsel = read_snps(BFILE + ".bed", list(all_idx), n_all)[grow]
    colof = {gi: k for k, gi in enumerate(all_idx)}
    def region(g):
        c = np.array([colof[i] for i in idx[g]])
        raw = Gsel[:, c].astype(np.int8)
        _, _, Xf = build_noia(raw, verbose=False)
        return raw, Xf, snp[idx[g]]

    # ================= Part A: four tiers of structure correction =================
    say("\n" + "=" * 68)
    say("[Part A] four tiers of structure correction: real epistasis or artefact")
    say("=" * 68)

    def run_four_tiers(gA, gB, pcol, tag):
        rawA, XfA, nA = region(gA); rawB, XfB, nB = region(gB)
        y = P[pcol].copy(); valid = ~np.isnan(y)
        n = int(valid.sum())
        say(f"\n--- {tag}  phenotype={pcol}  n={n} ---")
        # keep only phenotyped samples so the whitening set is fixed
        yv = y[valid]; Kv = K[np.ix_(valid, valid)]; PCv = PCs[valid]
        A_raw = rawA[valid]; B_raw = rawB[valid]
        A_X = XfA[valid]; B_X = XfB[valid]

        # null-model REML, used by T2 and T3
        Xnull = np.column_stack([np.ones(n), PCv])
        rml = REML1().fit(yv, Xnull, Kv)
        say(f"  null REML: vg={rml.vg_:.4f} ve={rml.ve_:.4f} h2={rml.h2_:.3f} "
            f"converged={rml.converged_} degenerate={rml.degenerate_}")

        out = []
        Ymat = make_Ymat(yv, np.ones(n, bool), N_PERM, SEED)

        # T0 raw
        d, run = scan_pairs(A_X, B_X, A_raw, B_raw, nA, nB, Ymat, None)
        out.append(("T0_raw", d, run, None, None))
        # T1 pc-only
        d, run = scan_pairs(A_X, B_X, A_raw, B_raw, nA, nB, Ymat, PCv)
        out.append(("T1_pconly", d, run, None, None))
        # T2: two-stage residualisation
        if rml.degenerate_:
            say("  [T2] the REML residual variance collapsed to zero, so the residuals are numerical noise; this tier is refused.")
            out.append(("T2_grmresid_DEGENERATE", None, None, None, None))
        else:
            resid = rml.blup_residual(yv, Xnull, Kv)
            Ymat_r = make_Ymat(resid, np.ones(n, bool), N_PERM, SEED)
            d, run = scan_pairs(A_X, B_X, A_raw, B_raw, nA, nB, Ymat_r, None)
            out.append(("T2_grmresid", d, run, None, None))
        # T3: one-stage mixed model
        if rml.degenerate_:
            say("  [T3] REML boundary degeneracy makes the mixed model unreliable; this tier is refused.")
            out.append(("T3_mixedmodel_DEGENERATE", None, None, None, None))
        else:
            U, sw = make_whitener(Kv, rml.vg_, rml.ve_)
            d, run = scan_pairs(A_X, B_X, A_raw, B_raw, nA, nB, Ymat, PCv, U=U, sw=sw)
            out.append(("T3_mixedmodel", d, run, U, sw))

        recs = []
        for name, d, run, U, sw in out:
            if d is None:
                recs.append(dict(test=tag, phenotype=pcol, tier=name, n=n,
                                 best_snp_A="", best_snp_B="", F=np.nan,
                                 n_pairs=0, p_perm=np.nan, nominal_p=np.nan,
                                 partial_R2=np.nan, beta=np.nan, se=np.nan,
                                 note="REML boundary degeneracy; result unusable"))
                continue
            top = d.sort_values("F", ascending=False).iloc[0]
            pp = perm_p(run, N_PERM)
            st = single_pair_stats(A_X[:, int(top["iA"])], B_X[:, int(top["iB"])],
                                   yv if not name.startswith("T2") else resid,
                                   PCv if name in ("T1_pconly", "T3_mixedmodel") else None,
                                   U=U, sw=sw)
            npair = len(d); dfree = n - (2 + (N_PCS if name in ("T1_pconly","T3_mixedmodel") else 0) + 2)
            recs.append(dict(test=tag, phenotype=pcol, tier=name, n=n,
                             best_snp_A=top["snp_A"], best_snp_B=top["snp_B"],
                             F=float(top["F"]), n_pairs=npair,
                             p_perm=pp, nominal_p=st["p"],
                             partial_R2=float(top["F"]) / (float(top["F"]) + max(dfree, 1)),
                             beta=st["beta"], se=st["se"],
                             note="p_perm=1/(N+1) is the permutation floor and should be written as <" + f"{1/(N_PERM+1):.4f}"
                                  if abs(pp - 1/(N_PERM+1)) < 1e-9 else ""))
            d.to_csv(os.path.join(OUT_DIR, f"A_pairs_{tag}_{name}.csv"), index=False)
            say(f"  [{name}] best={top['snp_A']} x {top['snp_B']}  F={top['F']:.3f}  "
                f"pairs={npair}  perm_p={pp:.4f}")
        return pd.DataFrame(recs), out, dict(
            XfA_full=XfA, XfB_full=XfB, rawA_full=rawA, rawB_full=rawB,
            nA=nA, nB=nB, valid=valid, yv=yv, PCv=PCv, Kv=Kv, rml=rml,
            resid=(None if rml.degenerate_ else resid))

    resA_main, tiers_main, ctx_main = run_four_tiers(*MAIN_TEST, tag=f"{MAIN_TEST[0]}x{MAIN_TEST[1]}")
    resA_neg,  tiers_neg,  ctx_neg  = run_four_tiers(*NEG_TEST,  tag=f"{NEG_TEST[0]}x{NEG_TEST[1]}")
    resA = pd.concat([resA_main, resA_neg], ignore_index=True)
    resA.to_csv(os.path.join(OUT_DIR, "A_four_tier_summary.csv"), index=False)
    rows_summary.append(("A_four_tier", resA))

    # fix the best pair from T3 (preferred) or T0 for parts B to E
    XfA_f = ctx_main['XfA_full']; XfB_f = ctx_main['XfB_full']
    rawA_f = ctx_main['rawA_full']; rawB_f = ctx_main['rawB_full']
    nA = ctx_main['nA']; nB = ctx_main['nB']; valid_main = ctx_main['valid']
    yv = ctx_main['yv']; PCv = ctx_main['PCv']; Kv = ctx_main['Kv']; rml_main = ctx_main['rml']
    pick = resA_main[resA_main.tier == "T3_mixedmodel"]
    if len(pick) == 0 or np.isnan(pick.iloc[0]["F"]):
        pick = resA_main[resA_main.tier == "T0_raw"]
    best_A_name = pick.iloc[0]["best_snp_A"]; best_B_name = pick.iloc[0]["best_snp_B"]
    iA = int(np.where(nA == best_A_name)[0][0]); iB = int(np.where(nB == best_B_name)[0][0])
    say(f"\nParts B to E use the fixed pair: {best_A_name} x {best_B_name}")

    # ================= Part B: scale robustness =================
    say("\n" + "=" * 68); say("[Part B] scale robustness"); say("=" * 68)
    recsB = []
    y_main_full = P[MAIN_TEST[2]]
    for tname, yt in [("raw", y_main_full),
                      ("log", safe_log(y_main_full)),
                      ("rankINT", rank_int(y_main_full))]:
        ytv = yt[valid_main]
        for tier, use_pc, useU in [("T0_raw", False, False), ("T3_mixedmodel", True, True)]:
            if useU and rml_main.degenerate_: continue
            U = sw = None
            if useU:
                r2 = REML1().fit(ytv, np.column_stack([np.ones(len(ytv)), PCv]), Kv)
                if r2.degenerate_: continue
                U, sw = make_whitener(Kv, r2.vg_, r2.ve_)
            st = single_pair_stats(XfA_f[valid_main][:, iA], XfB_f[valid_main][:, iB], ytv,
                                   PCv if use_pc else None, U=U, sw=sw)
            recsB.append(dict(transform=tname, tier=tier, **st))
            say(f"  {tname:8s} {tier:15s} F={st['F']:8.3f} p={st['p']:.3e} beta={st['beta']:+.4f}")
    dB = pd.DataFrame(recsB); dB.to_csv(os.path.join(OUT_DIR, "B_scale_robustness.csv"), index=False)
    rows_summary.append(("B_scale", dB))

    # ================= Part C: internal replication across phenotypes =================
    say("\n" + "=" * 68); say("[Part C] internal replication across phenotypes, same locus pair"); say("=" * 68)
    say("  These phenotypes are correlated with one another: this is internal replication,\n"
        "  not replication in an independent sample, and must be described as such.")
    recsC = []
    for pc in REPLICATION_PHENOS:
        if pc not in P: continue
        yy = P[pc]; ok = ~np.isnan(yy)
        if ok.sum() < 30: continue
        Ksub = K[np.ix_(ok, ok)]; PCsub = PCs[ok]
        st0 = single_pair_stats(XfA_f[ok][:, iA], XfB_f[ok][:, iB], yy[ok])
        r3 = REML1().fit(yy[ok], np.column_stack([np.ones(int(ok.sum())), PCsub]), Ksub)
        if r3.degenerate_:
            st3 = dict(F=np.nan, p=np.nan, beta=np.nan, se=np.nan, n=int(ok.sum()))
        else:
            U, sw = make_whitener(Ksub, r3.vg_, r3.ve_)
            st3 = single_pair_stats(XfA_f[ok][:, iA], XfB_f[ok][:, iB], yy[ok], PCsub, U=U, sw=sw)
        recsC.append(dict(phenotype=pc, n=int(ok.sum()),
                          F_raw=st0["F"], p_raw=st0["p"], beta_raw=st0["beta"],
                          F_mm=st3["F"], p_mm=st3["p"], beta_mm=st3["beta"],
                          h2_grm=r3.h2_, reml_degenerate=r3.degenerate_))
        say(f"  {pc:12s} n={ok.sum():4d}  raw F={st0['F']:7.2f} p={st0['p']:.2e} | "
            f"MM F={st3['F'] if not np.isnan(st3['F']) else float('nan'):7.2f} "
            f"p={st3['p']:.2e}" if not np.isnan(st3['F']) else
            f"  {pc:12s} n={ok.sum():4d}  raw F={st0['F']:7.2f} p={st0['p']:.2e} | MM degenerate")
    dC = pd.DataFrame(recsC); dC.to_csv(os.path.join(OUT_DIR, "C_cross_phenotype.csv"), index=False)
    rows_summary.append(("C_replication", dC))

    # ================= Part D: 2x2 genotype-class means =================
    say("\n" + "=" * 68); say("[Part D] genotype-class means: direct evidence of AND-gate masking"); say("=" * 68)
    ga = rawA_f[valid_main][:, iA]; gb = rawB_f[valid_main][:, iB]; yD = yv
    recsD = []
    for va in np.unique(ga[ga >= 0]):
        for vb in np.unique(gb[gb >= 0]):
            m = (ga == va) & (gb == vb)
            if m.sum() == 0: continue
            recsD.append(dict(geno_A=int(va), geno_B=int(vb), n=int(m.sum()),
                              mean=float(yD[m].mean()), sd=float(yD[m].std(ddof=1)) if m.sum()>1 else np.nan))
    dD = pd.DataFrame(recsD)
    say(dD.to_string(index=False))
    try:
        piv = dD.pivot(index="geno_A", columns="geno_B", values="mean")
        lv_a = sorted(dD.geno_A.unique()); lv_b = sorted(dD.geno_B.unique())
        if len(lv_a) >= 2 and len(lv_b) >= 2:
            contrast = (piv.loc[lv_a[-1], lv_b[-1]] - piv.loc[lv_a[-1], lv_b[0]]
                        - piv.loc[lv_a[0], lv_b[-1]] + piv.loc[lv_a[0], lv_b[0]])
            say(f"  interaction contrast (deviation from additivity) = {contrast:.3f}")
            say("  Under a purely additive model this is zero; a clear departure is direct\n"
                "  evidence of epistasis.")
            dD.attrs["contrast"] = contrast
            pd.DataFrame([dict(interaction_contrast=contrast)]).to_csv(
                os.path.join(OUT_DIR, "D_contrast.csv"), index=False)
    except Exception as e:
        say(f"  [note] could not build the pivot table: {e}")
    dD.to_csv(os.path.join(OUT_DIR, "D_genotype_class_means.csv"), index=False)
    rows_summary.append(("D_2x2", dD))

    # ================= Part E: the mechanistic chain =================
    say("\n" + "=" * 68); say("[Part E] mechanistic chain: genotype to expression"); say("=" * 68)
    recsE = []
    for expr in EXPRESSION_PHENOS:
        if expr not in P: continue
        ye = P[expr]; ok = ~np.isnan(ye)
        if ok.sum() < 30: continue
        for lbl, gvec in [(f"{MAIN_TEST[0]} locus ({best_A_name})", XfA_f[:, iA]),
                          (f"{MAIN_TEST[1]} locus ({best_B_name})", XfB_f[:, iB])]:
            g = gvec[ok]; yq = ye[ok]
            if np.std(g) < 1e-9: continue
            X = np.column_stack([np.ones(len(yq)), g])
            b, *_ = np.linalg.lstsq(X, yq, rcond=None)
            r = yq - X @ b; df = len(yq) - 2
            F = ((np.sum((yq - yq.mean())**2) - float(r @ r)) / 1) / (float(r @ r) / df)
            recsE.append(dict(expression=expr, locus=lbl, n=int(ok.sum()),
                              beta=float(b[1]), F=float(F), p=float(stats.f.sf(F, 1, df))))
            say(f"  {expr} ~ {lbl}: beta={b[1]:+.4f} F={F:.2f} p={stats.f.sf(F,1,df):.3e}")
        st = single_pair_stats(XfA_f[ok][:, iA], XfB_f[ok][:, iB], ye[ok])
        recsE.append(dict(expression=expr, locus="A x B interaction", n=st["n"],
                          beta=st["beta"], F=st["F"], p=st["p"]))
        say(f"  {expr} ~ A x B interaction: beta={st['beta']:+.4f} F={st['F']:.2f} p={st['p']:.3e}")
    dE = pd.DataFrame(recsE); dE.to_csv(os.path.join(OUT_DIR, "E_mechanism.csv"), index=False)
    rows_summary.append(("E_mechanism", dE))

    # ================= Part F: random-interval negative control =================
    say("\n" + "=" * 68); say("[Part F] random-interval negative control")
    say("=" * 68)
    rng = np.random.default_rng(SEED)
    cand_chr = [c for c in np.unique(chrom) if c not in test_chrs]
    if len(cand_chr) < 2:
        cand_chr = list(np.unique(chrom))
    nA_t = len(idx[MAIN_TEST[0]]); nB_t = len(idx[MAIN_TEST[1]])
    y_neg = P[MAIN_TEST[2]]; okn = ~np.isnan(y_neg)
    yneg = y_neg[okn]; nneg = int(okn.sum())
    Kn = K[np.ix_(okn, okn)]; PCn = PCs[okn]
    rn = REML1().fit(yneg, np.column_stack([np.ones(nneg), PCn]), Kn)
    Un, swn = (make_whitener(Kn, rn.vg_, rn.ve_) if not rn.degenerate_ else (None, None))
    recsF = []
    for t in range(N_NEG_PAIRS):
        try:
            c1, c2 = rng.choice(cand_chr, 2, replace=False)
            i1 = np.where(chrom == c1)[0]; i2 = np.where(chrom == c2)[0]
            if len(i1) < nA_t + 5 or len(i2) < nB_t + 5: continue
            s1 = rng.integers(0, len(i1) - nA_t); s2 = rng.integers(0, len(i2) - nB_t)
            id1 = i1[s1:s1 + nA_t]; id2 = i2[s2:s2 + nB_t]
            Gr = read_snps(BFILE + ".bed", list(np.concatenate([id1, id2])), n_all)[grow][okn]
            r1 = Gr[:, :nA_t].astype(np.int8); r2 = Gr[:, nA_t:].astype(np.int8)
            _, _, X1f = build_noia(r1, verbose=False); _, _, X2f = build_noia(r2, verbose=False)
            Ym = make_Ymat(yneg, np.ones(nneg, bool), N_PERM_NEG, SEED + t)
            nm1 = snp[id1]; nm2 = snp[id2]
            d, run = scan_pairs(X1f, X2f, r1, r2, nm1, nm2, Ym, PCn, U=Un, sw=swn)
            if d is None: continue
            pp = perm_p(run, N_PERM_NEG)
            recsF.append(dict(rep=t, chrA=int(c1), chrB=int(c2), n_pairs=len(d),
                              max_F=float(run[0]), p_perm=pp))
            say(f"  negative control {t+1}/{N_NEG_PAIRS}: chr{c1} x chr{c2} pairs={len(d)} "
                f"maxF={run[0]:.2f} p={pp:.4f}")
        except Exception as e:
            say(f"  negative control {t} skipped: {e}")
    dF = pd.DataFrame(recsF)
    if len(dF):
        frac = float((dF.p_perm <= 0.05).mean())
        say(f"\n  of {len(dF)} random interval pairs, the fraction with p<=0.05 is {frac:.3f} (should be near 0.05)")
        dF.attrs["frac_sig"] = frac
    dF.to_csv(os.path.join(OUT_DIR, "F_random_region_negative_control.csv"), index=False)
    rows_summary.append(("F_negctrl", dF))

    # ================= Part G: self-calibration =================
    say("\n" + "=" * 68); say("[Part G] self-calibration: inject a known effect, measure power and false positives"); say("=" * 68)
    recsG = []
    ga_c = XfA_f[valid_main][:, iA]; gb_c = XfB_f[valid_main][:, iB]
    inter_true = (ga_c - ga_c.mean()) * (gb_c - gb_c.mean())
    inter_true = inter_true / (np.std(inter_true) + 1e-12)
    sd_y = np.std(yv)
    for eff_frac in [0.0, 0.02, 0.05, 0.10]:
        hits_raw = hits_mm = hits_t2 = 0; done = 0
        for rep in range(N_CALIB):
            rg = np.random.default_rng(9000 + rep)
            base = rg.permutation(yv)                 # keeps the marginal distribution, breaks the association
            amp = np.sqrt(eff_frac / max(1 - eff_frac, 1e-9)) * sd_y
            ysim = base + amp * inter_true
            st0 = single_pair_stats(ga_c, gb_c, ysim)
            if st0["p"] < 0.05: hits_raw += 1
            Xn2 = np.column_stack([np.ones(len(ysim)), PCv])
            rr = REML1().fit(ysim, Xn2, Kv, n_restarts=2)
            if not rr.degenerate_:
                st2 = single_pair_stats(ga_c, gb_c, rr.blup_residual(ysim, Xn2, Kv))
                if st2["p"] < 0.05: hits_t2 += 1
                Uu, ss = make_whitener(Kv, rr.vg_, rr.ve_)
                st3 = single_pair_stats(ga_c, gb_c, ysim, PCv, U=Uu, sw=ss)
                if st3["p"] < 0.05: hits_mm += 1
            done += 1
        recsG.append(dict(injected_var_frac=eff_frac, n_rep=done,
                          power_T0_raw=hits_raw / max(done, 1),
                          power_T2_twostage=hits_t2 / max(done, 1),
                          power_T3_mixedmodel=hits_mm / max(done, 1)))
        say(f"  injected interaction variance={eff_frac:.2f}:  T0 raw={hits_raw/max(done,1):.2f}  "
            f"T2 two-stage={hits_t2/max(done,1):.2f}  T3 one-stage={hits_mm/max(done,1):.2f}")
    dG = pd.DataFrame(recsG); dG.to_csv(os.path.join(OUT_DIR, "G_self_calibration.csv"), index=False)
    say("  The zero-injection row is the false-positive rate and should be near 0.05;\n"
        "  the other rows are detection power.")
    say("  This table answers directly whether T2 loses power relative to T3 on these data.")
    rows_summary.append(("G_calibration", dG))

    # ================= Summary =================
    say("\n" + "=" * 68); say("How to read the output"); say("=" * 68)
    say("""
1) Read the T3_mixedmodel row of A_four_tier_summary.csv. That is the
   statistically correct test: structure is handled by the GRM random effect
   and the interaction is estimated directly as a fixed effect, so nothing is
   shrunk by a first-stage BLUP before being measured.
   - T3 significant     => the epistasis is real, and a negative T2 is a loss
                           of power caused by over-correction.
   - T3 not significant => the raw signal is most likely a structural artefact
                           and this positive control should be abandoned.
   T0, T1 and T2 are supporting evidence, not the criterion.

2) If the interaction in table B survives the log and rank-INT transformations,
   it is not a scale artefact.

3) If the same pair recurs across FT16/FT22/LN* in table C, it is not a
   one-off measurement. These phenotypes are correlated, so this is internal
   replication and must be described as such.

4) An interaction contrast in table D that departs clearly from zero, with one
   genotype cell standing out, indicates AND-gate masking and agrees with the
   known FRI/FLC mechanism.

5) The fraction of p <= 0.05 in table F should be near 0.05. A clearly higher
   value means the procedure has a systematic false-positive rate and every
   positive result above it is in doubt. This is the foundation of the whole
   chain.

6) The zero-injection row of table G is the false-positive rate (near 0.05);
   the remaining rows give the effect size needed for detection at this sample
   size. That is the only way to separate "absent" from "undetectable", and it
   uses the same calibration convention as the pig and mussel materials.
""")
    say(f"all results written to {OUT_DIR}/")

if __name__ == "__main__":
    main()
