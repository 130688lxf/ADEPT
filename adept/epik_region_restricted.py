#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPIK-NOIA restricted to gene intervals (Arabidopsis positive control)
===================================================================
The statistics are identical to the genome-wide version; only the search space
differs. Instead of "genome-wide NOIA screen -> candidate pool -> enumerate
within the pool", every pair between two specified gene intervals is enumerated.

Shared with the genome-wide version:
  - orthogonal NOIA coding Ha / Hd, with a hard orthogonality check
  - four model families AD / DD / MULT / XOR, all tested by nested-model F
  - an independent max-statistic permutation test per family
  - Bonferroni across families plus BH FDR
  - beta_interaction and its SE
  - a genotype-cell count filter. Joint cells are counted here, which is
    stricter than marginal cells and targets the case where a sparse
    double heterozygote class lets a handful of individuals determine the
    interaction term.

Structure correction specific to the interval-restricted version:
  - genome-wide additive GRM -> PCs as fixed effects, plus REML BLUP
    residualisation
  - the GRM EXCLUDES the chromosomes being tested (chr4, chr5) by default, so
    the FRI and FLC signals are not absorbed into the GRM and then subtracted
    again (proximal contamination)
  - both the residualised and the non-residualised results are written out for
    comparison

Two degeneracies of inbred data, detected and reported rather than hidden:
  - Arabidopsis accessions are homozygous, so the heterozygote frequency is
    near zero, the NOIA Hd is near zero, and the AD and DD families carry no
    information; they are skipped automatically.
  - On homozygous data Xx = (Xa>0)*2-1 equals Xa-1, a linear transformation, so
    conditional on the main effects the XOR and MULT F statistics are identical
    and are reported jointly.
===================================================================
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve, LinAlgError

# ----------------------- Configuration -----------------------
BFILE   = "genotype_bed"
PHENO   = "phenotypes.pheno"
OUT_DIR = "arabidopsis_epik_region_results"

TRAIT   = "FT10"          # main phenotype (continuous)
N_PERM  = 1000
MIN_MAF = 0.10
MIN_CELL = 5              # minimum count per joint genotype cell
WINDOW  = 20000
N_PCS   = 5
SEED    = 12345
GRM_CHUNK = 20000         # SNPs read per chunk when accumulating the GRM
EXCLUDE_CHR_FROM_GRM = [4, 5]   # tested chromosomes are kept out of the GRM

GENES = {
    "RPM1": (3, 2217000, 2221000),
    "RIN4": (3, 9127000, 9129000),
    "FRI" : (4,  268000,  270000),
    "FLC" : (5, 3173000, 3180000),
}
# interval pairs to run: (gene A, gene B, phenotype column, binary?)
TESTS = [
    ("FRI",  "FLC",  "FT10", False),
    ("RPM1", "RIN4", "<i>avrRpm1</i>", True),
]

os.makedirs(OUT_DIR, exist_ok=True)
rng_global = np.random.default_rng(SEED)

# =================== PLINK reading ===================
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
        assert f.read(3) == bytes([0x6c, 0x1b, 0x01])
        for r, idx in enumerate(idxs):
            f.seek(3 + idx * bps)
            b = np.frombuffer(f.read(bps), dtype=np.uint8)
            tb = np.stack([b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3],
                          axis=1).reshape(-1)[:n]
            out[r] = _LUT[tb]
    return out.T

# =================== Phenotypes and identifiers ===================
def load_pheno(path, cols):
    with open(path) as f:
        header = f.readline().split()
        rows = [l.split() for l in f]
    iid = [r[1] for r in rows]
    data = {}
    for name in cols:
        j = header.index(name)
        v = []
        for r in rows:
            try:
                fv = float(r[j]); v.append(fv if not np.isnan(fv) else np.nan)
            except Exception:
                v.append(np.nan)
        data[name] = np.array(v)
    return iid, data

def match_ids(fam_iid, phe_iid):
    fmap = {v: i for i, v in enumerate(phe_iid)}
    pairs = [(gi, fmap[k]) for gi, k in enumerate(fam_iid) if k in fmap]
    if len(pairs) < 0.5 * len(fam_iid):
        def nz(s):
            s = str(s).strip(); return s[:-2] if s.endswith(".0") else s
        fmap = {nz(v): i for i, v in enumerate(phe_iid)}
        pairs = [(gi, fmap[nz(k)]) for gi, k in enumerate(fam_iid) if nz(k) in fmap]
    print(f"  {len(pairs)} identifier(s) matched")
    return pairs

# =================== Genome-wide additive GRM, in chunks ===================
def compute_grm_chunked(bedpath, n_samples, snp_idx_use, chunk=GRM_CHUNK):
    """VanRaden additive GRM accumulated in chunks, so 210k SNPs never sit in memory at once."""
    G = np.zeros((n_samples, n_samples), dtype=np.float64)
    denom = 0.0
    total = len(snp_idx_use)
    for s in range(0, total, chunk):
        idxs = snp_idx_use[s:s + chunk]
        Xc = read_snps(bedpath, list(idxs), n_samples).astype(np.float32)
        miss = Xc < 0
        if miss.any():
            col_mean = np.array([Xc[~miss[:, k], k].mean() if (~miss[:, k]).any() else 0.0
                                 for k in range(Xc.shape[1])], dtype=np.float32)
            Xc[miss] = np.take(col_mean, np.where(miss)[1])
        p = Xc.mean(axis=0) / 2.0
        keep = (p > 0.005) & (p < 0.995)
        Xc = Xc[:, keep]; p = p[keep]
        if Xc.shape[1] == 0:
            continue
        Z = Xc - 2 * p
        G += (Z @ Z.T).astype(np.float64)
        denom += float(2 * np.sum(p * (1 - p)))
        print(f"    GRM progress {min(s+chunk,total)}/{total}", flush=True)
    return G / max(denom, 1e-12)

def pcs_from_grm(K, n_pcs):
    w, V = np.linalg.eigh(K)
    o = np.argsort(w)[::-1][:n_pcs]
    return V[:, o] * np.sqrt(np.maximum(w[o], 0.0))

# =================== REML ===================
class OneKernelREML:
    def __init__(self, jitter=1e-6): self.jitter = jitter
    def _neg2(self, lp, y, X, K):
        n = len(y); v = np.exp(lp)
        V = v[0] * K + v[1] * np.eye(n) + self.jitter * np.eye(n)
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
            init = np.log(np.maximum(max(vy / 2, 1e-4) * sc, 1e-4))
            res = minimize(self._neg2, init, args=(y, X, K), method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5, "adaptive": True})
            if best is None or res.fun < best.fun: best = res
        self.v_ = np.exp(best.x); self.converged_ = best.success; self.res_ = best
        return self
    def residual(self, y, X, K):
        n = len(y); v = self.v_
        V = v[0] * K + v[1] * np.eye(n) + self.jitter * np.eye(n)
        L = cholesky(V, lower=True)
        Viy = cho_solve((L, True), y); ViX = cho_solve((L, True), X)
        b = np.linalg.solve(X.T @ ViX, X.T @ Viy)
        r = y - X @ b
        u = v[0] * (K @ cho_solve((L, True), r))
        return y - X @ b - u

# =================== NOIA coding ===================
def build_noia(X, tol=1e-6):
    X = X.astype(np.float64)
    Xf = X.copy()
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
    H = (Xf == 1).astype(np.float64)
    beta = (p_Aa * (1 - mu)) / np.where(mono, 1.0, varA)
    Hd = H - p_Aa - beta * Ha
    Hd[:, mono] = 0.0
    ma = np.abs(Ha.mean(0)).max(); md = np.abs(Hd.mean(0)).max()
    cad = np.abs((Ha * Hd).mean(0) - Ha.mean(0) * Hd.mean(0)).max()
    print(f"    NOIA orthogonality: max|E(A)|={ma:.2e}, max|E(D)|={md:.2e}, max|Cov(A,D)|={cad:.2e}")
    if ma >= tol or md >= tol or cad >= tol:
        raise AssertionError("NOIA orthogonality check failed")
    return Ha, Hd, Xf

# ========= Vectorised interaction F, permutations as a matrix dimension =========
def pairF_vec(v1, v2, Ymat, valid):
    nm = int(valid.sum())
    if nm < 10: return None
    a = v1[valid]; b = v2[valid]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9: return None
    inter = a * b
    if np.std(inter) < 1e-9: return None
    Y = Ymat[valid]
    X0 = np.column_stack([np.ones(nm), a, b]); X1 = np.column_stack([X0, inter])
    Q0, _ = np.linalg.qr(X0); Q1, R1 = np.linalg.qr(X1)
    if np.min(np.abs(np.diag(R1))) < 1e-8: return None
    df = nm - 4
    if df <= 0: return None
    SS = (Y**2).sum(0)
    R0 = SS - ((Q0.T @ Y)**2).sum(0)
    R1s = np.maximum(SS - ((Q1.T @ Y)**2).sum(0), 1e-300)
    return ((R0 - R1s) / 1.0) / (R1s / df)

def effect_size(v1, v2, y, valid):
    a = v1[valid]; b = v2[valid]; yy = y[valid]
    X = np.column_stack([np.ones(len(yy)), a, b, a * b])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    r = yy - X @ beta; df = len(yy) - 4
    if df <= 0: return beta[-1], np.nan
    s2 = np.sum(r**2) / df
    try:
        se = np.sqrt(max(s2 * np.linalg.inv(X.T @ X)[-1, -1], 0.0))
    except np.linalg.LinAlgError:
        se = np.nan
    return beta[-1], se

# =================== Filtering ===================
def maf_ok(g):
    gg = g[g >= 0]
    if len(gg) == 0: return False
    p = gg.mean() / 2.0
    return min(p, 1 - p) >= MIN_MAF

def joint_cells_ok(gi, gj, valid):
    a = gi[valid]; b = gj[valid]
    combo = a * 3 + b
    _, cnt = np.unique(combo, return_counts=True)
    return len(cnt) >= 4 and (cnt >= MIN_CELL).all()

# =================== One interval pair x one model family ===================
def scan_family(fam, VA, VB, Graw_A, Graw_B, nameA, nameB, Ymat, base_valid, n_perm):
    """Returns (per-pair DataFrame, vector of the maximum F per permutation), or None if the family is degenerate."""
    if np.abs(VA).max() < 1e-9 or np.abs(VB).max() < 1e-9:
        print(f"    [{fam}] the coding is identically zero (an inbred panel has no heterozygotes, so NOIA Hd degenerates); this family carries no information and is skipped.")
        return None, None
    running = np.full(Ymat.shape[1], -np.inf)
    rows = []
    n_filt = 0
    for i in range(VA.shape[1]):
        gi_raw = Graw_A[:, i]
        if not maf_ok(gi_raw): continue
        for j in range(VB.shape[1]):
            gj_raw = Graw_B[:, j]
            if not maf_ok(gj_raw): continue
            valid = base_valid & (gi_raw >= 0) & (gj_raw >= 0)
            if not joint_cells_ok(gi_raw, gj_raw, valid):
                n_filt += 1; continue
            F = pairF_vec(VA[:, i], VB[:, j], Ymat, valid)
            if F is None: continue
            np.maximum(running, F, out=running)
            rows.append((nameA[i], nameB[j], i, j, F[0]))
    if not rows:
        print(f"    [{fam}] no eligible pairs ({n_filt} removed by the joint-cell filter).")
        return None, None
    df = pd.DataFrame(rows, columns=["snp_A", "snp_B", "iA", "iB", "F"])
    print(f"    [{fam}] tested {len(df)} pairs ({n_filt} removed by the joint-cell filter), max F={running[0]:.3f}")
    return df, running

def run_region_pair(gA, gB, y, tag, Gsel, colof, idx, snp, n_perm=N_PERM):
    base_valid = ~np.isnan(y)
    n = int(base_valid.sum())
    print(f"\n  --- {tag}  (n={n}) ---")
    cA = np.array([colof[i] for i in idx[gA]]); cB = np.array([colof[i] for i in idx[gB]])
    XA = Gsel[:, cA].astype(np.int8); XB = Gsel[:, cB].astype(np.int8)
    nameA = snp[idx[gA]]; nameB = snp[idx[gB]]

    HaA, HdA, XfA = build_noia(XA)
    HaB, HdB, XfB = build_noia(XB)
    het = float(np.mean(XfA == 1))
    print(f"    heterozygote frequency in interval A={het:.4f}  -> "
          f"{'Hd degenerate, AD/DD carry no information' if het<1e-3 else 'Hd usable'}")

    XxA = (XfA > 0).astype(float) * 2 - 1
    XxB = (XfB > 0).astype(float) * 2 - 1

    yv = y.copy(); yv[~base_valid] = 0.0
    Ymat = np.empty((len(y), n_perm + 1))
    Ymat[:, 0] = yv
    rng = np.random.default_rng(SEED)
    vidx = np.where(base_valid)[0]
    for k in range(n_perm):
        col = np.zeros(len(y)); col[vidx] = rng.permutation(y[vidx])
        Ymat[:, k + 1] = col

    fams = {
        "MULT": (XfA, XfB),
        "XOR":  (XxA, XxB),
        "AD":   (HaA, HdB),
        "DD":   (HdA, HdB),
    }
    out, nulls = {}, {}
    for fam, (VA, VB) in fams.items():
        d, run = scan_family(fam, VA, VB, XA, XB, nameA, nameB, Ymat, base_valid, n_perm)
        if d is not None:
            out[fam] = d; nulls[fam] = run

    if not out:
        print("    every family is degenerate; no result."); return None

    # an independent max-statistic permutation p per family
    recs = []
    for fam, d in out.items():
        run = nulls[fam]
        obs = run[0]
        p_perm = (1 + int(np.sum(run[1:] >= obs))) / (n_perm + 1)
        top = d.sort_values("F", ascending=False).iloc[0]
        recs.append({"region_pair": tag, "family": fam,
                     "best_snp_A": top["snp_A"], "best_snp_B": top["snp_B"],
                     "best_F": obs, "n_pairs": len(d),
                     "p_perm_family": p_perm, "iA": int(top["iA"]), "iB": int(top["iB"])})
    res = pd.DataFrame(recs)
    # Bonferroni across families, counting non-degenerate families only, plus BH FDR
    nfam = len(res)
    res["p_bonf_across_families"] = np.minimum(res["p_perm_family"] * nfam, 1.0)
    order = np.argsort(res["p_bonf_across_families"].values)
    ranked = np.empty(nfam); ranked[order] = np.arange(1, nfam + 1)
    res["FDR_BH"] = np.minimum(1.0, res["p_bonf_across_families"].values * nfam / ranked)

    # effect sizes
    bs, ss = [], []
    for _, r in res.iterrows():
        VA, VB = fams[r["family"]]
        gi = XA[:, int(r["iA"])]; gj = XB[:, int(r["iB"])]
        valid = base_valid & (gi >= 0) & (gj >= 0)
        b, se = effect_size(VA[:, int(r["iA"])], VB[:, int(r["iB"])], np.nan_to_num(y), valid)
        bs.append(b); ss.append(se)
    res["beta_interaction"] = bs; res["se_interaction"] = ss
    for fam, d in out.items():
        d.to_csv(os.path.join(OUT_DIR, f"pairs_{tag}_{fam}.csv"), index=False)
    return res

# ============================ Main ============================
def main():
    print("reading .bim / .fam ...")
    chrom, snp, pos = read_bim(BFILE + ".bim")
    fam_iid = read_fam(BFILE + ".fam")
    n_samples = len(fam_iid)
    print(f"  {len(snp)} SNPs, {n_samples} samples")

    phe_cols = sorted({t[2] for t in TESTS})
    phe_iid, phe = load_pheno(PHENO, phe_cols)
    pairs = match_ids(fam_iid, phe_iid)
    grow = np.array([p[0] for p in pairs]); prow = np.array([p[1] for p in pairs])

    idx = {k: np.where((chrom == v[0]) & (pos >= v[1] - WINDOW) & (pos <= v[2] + WINDOW))[0]
           for k, v in GENES.items()}
    print("  SNPs per interval: " + ", ".join(f"{k}={len(v)}" for k, v in idx.items()))
    all_idx = np.unique(np.concatenate(list(idx.values())))
    Gsel_full = read_snps(BFILE + ".bed", list(all_idx), n_samples)
    Gsel = Gsel_full[grow]
    colof = {gi: k for k, gi in enumerate(all_idx)}

    print(f"\ncomputing the genome-wide additive GRM, excluding chr{EXCLUDE_CHR_FROM_GRM} to avoid proximal contamination ...")
    use = np.where(~np.isin(chrom, EXCLUDE_CHR_FROM_GRM))[0]
    print(f"  SNPs entering the GRM: {len(use)}")
    K_full = compute_grm_chunked(BFILE + ".bed", n_samples, use)
    K = K_full[np.ix_(grow, grow)]
    PCs_all = pcs_from_grm(K, N_PCS)

    all_res = []
    for gA, gB, pcol, is_bin in TESTS:
        y_raw = phe[pcol][prow]
        valid = ~np.isnan(y_raw)
        tag = f"{gA}x{gB}"

        # ---- (a) not residualised ----
        r1 = run_region_pair(gA, gB, y_raw, f"{tag}_raw", Gsel, colof, idx, snp)
        if r1 is not None:
            r1["phenotype"] = pcol; r1["correction"] = "none"; all_res.append(r1)

        # ---- (b) residualised on the GRM and PCs ----
        Ks = K[np.ix_(valid, valid)]
        Xd = np.column_stack([np.ones(int(valid.sum())), PCs_all[valid]])
        m = OneKernelREML().fit(y_raw[valid], Xd, Ks)
        vg, ve = m.v_
        print(f"\n  [{tag}] REML: sigma_g2={vg:.4f}, sigma_e2={ve:.4f}, "
              f"h2={vg/(vg+ve):.3f}, converged={m.converged_}")
        resid = np.full(len(y_raw), np.nan)
        resid[valid] = m.residual(y_raw[valid], Xd, Ks)
        r2 = run_region_pair(gA, gB, resid, f"{tag}_grmresid", Gsel, colof, idx, snp)
        if r2 is not None:
            r2["phenotype"] = pcol; r2["correction"] = "GRM_BLUP+PC"; all_res.append(r2)

    if all_res:
        final = pd.concat(all_res, ignore_index=True)
        cols = ["region_pair", "phenotype", "correction", "family", "best_snp_A", "best_snp_B",
                "best_F", "n_pairs", "p_perm_family", "p_bonf_across_families", "FDR_BH",
                "beta_interaction", "se_interaction"]
        final = final[cols]
        final.to_csv(os.path.join(OUT_DIR, "00_epik_region_summary.csv"), index=False)
        print("\n" + "=" * 70)
        print(final.to_string(index=False))
        print("=" * 70)
        print(f"\nresults written to {OUT_DIR}/")

if __name__ == "__main__":
    main()
