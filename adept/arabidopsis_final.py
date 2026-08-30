#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabidopsis validation, final stage: pre-specified loci, a comparison of
correction schemes, and a decisive injection test
====================================================================
Run in the same directory as arabidopsis_full_validation.py, from which the
underlying functions are imported rather than reimplemented.

Four questions.

[Q1] With the loci specified in advance, is the interaction still significant?
  The hypothesis "FRI interacts with FLC" comes from molecular biology and is
  specified in advance, not mined from the data. There is therefore one test,
  which should not carry a 6470-fold multiple-testing penalty. The loci are
  chosen by rules BLIND to the interaction statistic, three of them as mutual
  robustness checks:
    R1  the SNP nearest the centre of the gene
    R2  the SNP with the strongest main effect in the interval (a different
        statistic)
    R3  the SNP with the highest MAF in the interval (ignores the phenotype
        entirely)
  This is what makes the word "passed" mean anything.

[Q2] Is the structure correction excessive?
  Fitting five PCs as fixed effects together with the full GRM as a random
  effect corrects the leading structural axes twice, because the PCs are
  computed from the GRM. GEMMA and EMMAX use one or the other. Four schemes
  are compared side by side:
    C0 none / C1 PCs only / C2 GRM only (mixed model) / C3 PCs + GRM
  If C3 is clearly weaker than C2, that is direct evidence of double
  correction.

[Q3] A fixed ruler: does the MAIN effect of FRI survive correction?
  FRI is cloned and confirmed by transgenic work, and its main effect on
  flowering time is not in dispute. A correction tier that kills even this
  established true positive is over-correcting, and its verdict on the
  interaction cannot be trusted. The ruler is fully determined and leaves no
  room for interpretation.

[Q4] The decisive test: inject a known interaction at the real loci and see
  which tiers detect it.
  The background must retain the real structural covariance, u~N(0, vg*K) +
  e~N(0, ve*I). Generating the background by permuting y is wrong: permutation
  destroys the structure as well, the tiers then perform alike, and the
  comparison cannot separate over-correction from a structural artefact.
    injection 0   -> the false-positive rate. If the uncorrected tier is
                     clearly above 0.05, structure-correlated loci do fabricate
                     interactions, which supports the confounding explanation.
    injection >0  -> power. If the corrected tiers fall clearly below the
                     uncorrected one, correction is killing real signal, which
                     supports the over-correction explanation.
  Both explanations are read off the same experiment.

A direct measure of confounding is also written out: the R2 of the interaction
vector regressed on the PCs.
====================================================================
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arabidopsis_full_validation import (
    read_bim, read_fam, read_snps, load_pheno_all, match_ids,
    compute_grm_chunked, pcs_from_grm, REML1, make_whitener,
    single_pair_stats, build_noia, rank_int, safe_log, GENES, say
)

# ---------------------------- Configuration ----------------------------
BFILE   = "genotype_bed"
PHENO   = "phenotypes.pheno"
OUT_DIR = "arabidopsis_final"
GENE_A, GENE_B = "FRI", "FLC"
MAIN_PHENO = "FT10"
REPL_PHENOS = ["FT10", "FT16", "FT22", "LN10", "LN16", "LN22", "FT_Field", "FT_GH"]
WINDOW  = 20000
N_PCS   = 5
MIN_MAF = 0.10
MIN_CELL = 5
N_INJECT = 100          # replicates per injection level
INJECT_FRACS = [0.0, 0.02, 0.05, 0.10]
SEED = 12345

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------- Correction schemes ----------------------
def build_scheme(name, y, PC, K):
    """Return (covars, U, sw). U=None means no mixed-model whitening."""
    n = len(y)
    if name == "C0_none":
        return None, None, None, dict(h2=np.nan, degenerate=False)
    if name == "C1_PConly":
        return PC, None, None, dict(h2=np.nan, degenerate=False)
    Xn = np.ones((n, 1)) if name == "C2_GRMonly" else np.column_stack([np.ones(n), PC])
    r = REML1().fit(y, Xn, K, n_restarts=3)
    if r.degenerate_:
        return None, None, None, dict(h2=r.h2_, degenerate=True)
    U, sw = make_whitener(K, r.vg_, r.ve_)
    cov = None if name == "C2_GRMonly" else PC
    return cov, U, sw, dict(h2=r.h2_, degenerate=False)

SCHEMES = ["C0_none", "C1_PConly", "C2_GRMonly", "C3_PC_plus_GRM"]

def main_effect_test(g, y, covars=None, U=None, sw=None):
    """Single-locus main-effect test, used as the fixed ruler."""
    ok = ~np.isnan(y); n = int(ok.sum())
    base = np.ones((n, 1)) if covars is None else np.column_stack([np.ones(n), covars[ok]])
    gg = g[ok]; yy = y[ok]
    if U is not None:
        from arabidopsis_full_validation import transform
        base = transform(base, U, sw); gg = transform(gg, U, sw); yy = transform(yy, U, sw)
    X1 = np.column_stack([base, gg])
    b, *_ = np.linalg.lstsq(X1, yy, rcond=None); r1 = yy - X1 @ b
    b0, *_ = np.linalg.lstsq(base, yy, rcond=None); r0 = yy - base @ b0
    df = n - X1.shape[1]
    if df <= 0 or float(r1 @ r1) <= 0: return dict(F=np.nan, p=np.nan, beta=np.nan, n=n)
    F = ((float(r0 @ r0) - float(r1 @ r1)) / 1.0) / (float(r1 @ r1) / df)
    return dict(F=F, p=float(stats.f.sf(F, 1, df)), beta=float(b[-1]), n=n)

# ============================ Main ============================
def main():
    say("=" * 66); say("Arabidopsis validation, final stage"); say("=" * 66)

    chrom, snp, pos = read_bim(BFILE + ".bim")
    fam_iid = read_fam(BFILE + ".fam"); n_all = len(fam_iid)
    say(f"[loaded] {len(snp)} SNPs, {n_all} samples")

    wanted = sorted(set(REPL_PHENOS + [MAIN_PHENO]))
    phe_iid, phe, found = load_pheno_all(PHENO, wanted)
    pr = match_ids(fam_iid, phe_iid)
    grow = np.array([p[0] for p in pr]); prow = np.array([p[1] for p in pr])
    P = {k: v[prow] for k, v in phe.items()}

    idx = {g: np.where((chrom == GENES[g][0]) &
                       (pos >= GENES[g][1] - WINDOW) & (pos <= GENES[g][2] + WINDOW))[0]
           for g in (GENE_A, GENE_B)}
    say(f"  intervals: {GENE_A}={len(idx[GENE_A])}, {GENE_B}={len(idx[GENE_B])}")

    test_chrs = sorted({GENES[GENE_A][0], GENES[GENE_B][0]})
    say(f"[GRM] excluding chr{test_chrs}")
    use = np.where(~np.isin(chrom, test_chrs))[0]
    K = compute_grm_chunked(BFILE + ".bed", n_all, use)[np.ix_(grow, grow)]
    PC = pcs_from_grm(K, N_PCS)

    all_idx = np.unique(np.concatenate([idx[GENE_A], idx[GENE_B]]))
    Gsel = read_snps(BFILE + ".bed", list(all_idx), n_all)[grow]
    colof = {gi: k for k, gi in enumerate(all_idx)}
    def reg(g):
        c = np.array([colof[i] for i in idx[g]])
        raw = Gsel[:, c].astype(np.int8)
        _, _, Xf = build_noia(raw, verbose=False)
        return raw, Xf, snp[idx[g]], pos[idx[g]]
    rawA, XA, nmA, poA = reg(GENE_A)
    rawB, XB, nmB, poB = reg(GENE_B)

    y0 = P[MAIN_PHENO]; ok0 = ~np.isnan(y0); n0 = int(ok0.sum())
    say(f"  main phenotype {MAIN_PHENO}: n={n0}")

    # ---------- Q1: three locus rules, all blind to the interaction ----------
    say("\n" + "=" * 66)
    say("[Q1] loci specified in advance; none of the three rules looks at the interaction statistic")
    say("=" * 66)

    def pick(rawR, XR, nmR, poR, gene, rule):
        mid = (GENES[gene][1] + GENES[gene][2]) / 2.0
        cand = [i for i in range(XR.shape[1])
                if MIN_MAF <= min(rawR[rawR[:, i] >= 0, i].mean() / 2,
                                  1 - rawR[rawR[:, i] >= 0, i].mean() / 2)]
        if not cand: cand = list(range(XR.shape[1]))
        if rule == "R1_nearest":
            return cand[int(np.argmin([abs(poR[i] - mid) for i in cand]))]
        if rule == "R2_topmain":
            Fs = [main_effect_test(XR[:, i], y0)["F"] for i in cand]
            Fs = [f if np.isfinite(f) else -1 for f in Fs]
            return cand[int(np.argmax(Fs))]
        mafs = [min(rawR[rawR[:, i] >= 0, i].mean() / 2,
                    1 - rawR[rawR[:, i] >= 0, i].mean() / 2) for i in cand]
        return cand[int(np.argmax(mafs))]

    picks = {}
    for rule in ["R1_nearest", "R2_topmain", "R3_maxMAF"]:
        ia = pick(rawA, XA, nmA, poA, GENE_A, rule)
        ib = pick(rawB, XB, nmB, poB, GENE_B, rule)
        picks[rule] = (ia, ib)
        say(f"  {rule}: {nmA[ia]} ({int(abs(poA[ia]-(GENES[GENE_A][1]+GENES[GENE_A][2])/2))}bp from {GENE_A}) "
            f"x {nmB[ib]} ({int(abs(poB[ib]-(GENES[GENE_B][1]+GENES[GENE_B][2])/2))}bp from {GENE_B})")

    # ---------- Q2: four correction schemes x three rules ----------
    say("\n" + "=" * 66)
    say("[Q2] four correction schemes; a single-pair test needs no scan-level correction")
    say("=" * 66)
    rows = []
    yv = y0[ok0]; Kv = K[np.ix_(ok0, ok0)]; PCv = PC[ok0]
    for rule, (ia, ib) in picks.items():
        gA = XA[ok0][:, ia]; gB = XB[ok0][:, ib]
        for sch in SCHEMES:
            cov, U, sw, info = build_scheme(sch, yv, PCv, Kv)
            if info["degenerate"]:
                rows.append(dict(rule=rule, scheme=sch, **{k: np.nan for k in
                            ["F", "p", "beta", "se"]}, n=n0, h2=info["h2"],
                            note="REML boundary degeneracy")); continue
            st = single_pair_stats(gA, gB, yv, cov, U=U, sw=sw)
            rows.append(dict(rule=rule, scheme=sch, F=st["F"], p=st["p"],
                             beta=st["beta"], se=st["se"], n=st["n"],
                             h2=info["h2"], note=""))
            say(f"  {rule:12s} {sch:15s} F={st['F']:7.2f} p={st['p']:.3e} "
                f"beta={st['beta']:+.3f} (h2={info['h2']:.3f})")
    dQ2 = pd.DataFrame(rows); dQ2.to_csv(os.path.join(OUT_DIR, "Q2_correction_schemes.csv"), index=False)

    # ---------- Q3: the fixed ruler, the FRI main effect ----------
    say("\n" + "=" * 66)
    say("[Q3] fixed ruler: does the FRI main effect survive each correction")
    say("  (the FRI main effect on flowering time is confirmed by transgenic work; a tier\n"
        "   that removes it is over-correcting)")
    say("=" * 66)
    rows3 = []
    for rule, (ia, ib) in picks.items():
        for gene, X_, i_, nm_ in [(GENE_A, XA, ia, nmA), (GENE_B, XB, ib, nmB)]:
            g = X_[ok0][:, i_]
            for sch in SCHEMES:
                cov, U, sw, info = build_scheme(sch, yv, PCv, Kv)
                if info["degenerate"]: continue
                st = main_effect_test(g, yv, cov, U, sw)
                rows3.append(dict(rule=rule, gene=gene, snp=nm_[i_], scheme=sch,
                                  F=st["F"], p=st["p"], beta=st["beta"]))
        say(f"  --- {rule} ---")
        sub = [r for r in rows3 if r["rule"] == rule]
        for r in sub:
            say(f"    {r['gene']:4s} {r['scheme']:15s} F={r['F']:8.2f} p={r['p']:.3e}")
    dQ3 = pd.DataFrame(rows3); dQ3.to_csv(os.path.join(OUT_DIR, "Q3_main_effect_ruler.csv"), index=False)

    # ---------- confounding: how much of the interaction vector structure explains ----------
    say("\n" + "=" * 66)
    say("[confounding] R2 of the interaction vector on the PCs; higher means more collinear with structure")
    say("=" * 66)
    rows4 = []
    for rule, (ia, ib) in picks.items():
        a = XA[ok0][:, ia]; b = XB[ok0][:, ib]
        inter = (a - a.mean()) * (b - b.mean())
        Xp = np.column_stack([np.ones(n0), PCv])
        bb, *_ = np.linalg.lstsq(Xp, inter, rcond=None)
        r2 = 1 - np.sum((inter - Xp @ bb) ** 2) / max(np.sum((inter - inter.mean()) ** 2), 1e-12)
        r2a = _r2_on(a, Xp); r2b = _r2_on(b, Xp)
        rows4.append(dict(rule=rule, R2_interaction_on_PC=r2,
                          R2_locusA_on_PC=r2a, R2_locusB_on_PC=r2b))
        say(f"  {rule:12s} interaction R2={r2:.3f} | {GENE_A} locus R2={r2a:.3f} | {GENE_B} locus R2={r2b:.3f}")
    pd.DataFrame(rows4).to_csv(os.path.join(OUT_DIR, "Q_confounding_R2.csv"), index=False)

    # ---------- replication across phenotypes, using the R1 loci ----------
    say("\n" + "=" * 66)
    say("[replication] the pre-specified loci across several flowering phenotypes")
    say("=" * 66)
    ia, ib = picks["R1_nearest"]
    rows5 = []
    for pc in REPL_PHENOS:
        if pc not in P: continue
        yy = P[pc]; ok = ~np.isnan(yy)
        if ok.sum() < 30: continue
        Ks = K[np.ix_(ok, ok)]; PCs_ = PC[ok]
        out = {}
        for sch in ["C0_none", "C2_GRMonly"]:
            cov, U, sw, info = build_scheme(sch, yy[ok], PCs_, Ks)
            if info["degenerate"]: out[sch] = dict(F=np.nan, p=np.nan); continue
            out[sch] = single_pair_stats(XA[ok][:, ia], XB[ok][:, ib], yy[ok], cov, U=U, sw=sw)
        rows5.append(dict(phenotype=pc, n=int(ok.sum()),
                          F_none=out["C0_none"]["F"], p_none=out["C0_none"]["p"],
                          F_GRM=out["C2_GRMonly"]["F"], p_GRM=out["C2_GRMonly"]["p"]))
        say(f"  {pc:10s} n={ok.sum():4d}  uncorrected F={out['C0_none']['F']:7.2f} p={out['C0_none']['p']:.2e} "
            f"| GRM only F={out['C2_GRMonly']['F']:7.2f} p={out['C2_GRMonly']['p']:.2e}")
    dR = pd.DataFrame(rows5)
    if len(dR):
        k = int(dR.p_GRM.notna().sum())
        say(f"  Bonferroni over {k} phenotypes: smallest p_GRM x {k} = "
            f"{dR.p_GRM.min()*k:.3e}")
    dR.to_csv(os.path.join(OUT_DIR, "Q_replication.csv"), index=False)

    # ---------- Q4: the decisive injection test ----------
    say("\n" + "=" * 66)
    say("[Q4] decisive test: a known interaction injected at the real loci, on a background that keeps the real structural covariance")
    say("=" * 66)
    ia, ib = picks["R1_nearest"]
    a = XA[ok0][:, ia]; b = XB[ok0][:, ib]
    inter = (a - a.mean()) * (b - b.mean())
    inter = inter / (inter.std() + 1e-12)
    Xn = np.column_stack([np.ones(n0), PCv])
    r_null = REML1().fit(yv, Xn, Kv, n_restarts=3)
    say(f"  REML on the real phenotype: vg={r_null.vg_:.3f} ve={r_null.ve_:.3f} h2={r_null.h2_:.3f}")
    L = np.linalg.cholesky(Kv + 1e-6 * np.eye(n0))
    rows6 = []
    for frac in INJECT_FRACS:
        hits = {s: 0 for s in SCHEMES}; done = 0
        for rep in range(N_INJECT):
            rg = np.random.default_rng(20000 + rep)
            u = np.sqrt(r_null.vg_) * (L @ rg.normal(0, 1, n0))   # keeps the real structural covariance
            e = np.sqrt(r_null.ve_) * rg.normal(0, 1, n0)
            ybase = u + e
            amp = np.sqrt(frac / max(1 - frac, 1e-9)) * np.std(ybase)
            ysim = ybase + amp * inter
            for sch in SCHEMES:
                cov, U, sw, info = build_scheme(sch, ysim, PCv, Kv)
                if info["degenerate"]: continue
                st = single_pair_stats(a, b, ysim, cov, U=U, sw=sw)
                if np.isfinite(st["p"]) and st["p"] < 0.05: hits[sch] += 1
            done += 1
        rec = dict(injected_frac=frac, n_rep=done,
                   **{f"rate_{s}": hits[s] / max(done, 1) for s in SCHEMES})
        rows6.append(rec)
        say(f"  injection={frac:.2f}: " + "  ".join(
            f"{s.split('_')[0]}={hits[s]/max(done,1):.2f}" for s in SCHEMES))
    dI = pd.DataFrame(rows6); dI.to_csv(os.path.join(OUT_DIR, "Q4_injection.csv"), index=False)

    say("""
  How to read this, which is the verdict on the whole question:
    The injection=0.00 row is the FALSE-POSITIVE RATE.
      If C0 (no correction) is clearly above 0.05 while C2 (GRM only) is near
      0.05, structure-correlated loci really do fabricate interactions and the
      raw F carries a confounded component.
      If every tier is near 0.05, there is no mechanism by which structure
      fabricates interactions, and the raw signal is more likely to be real.
    The rows with injection > 0 are POWER.
      If C2 and C3 fall clearly below C0, correction is killing real signal
      (over-correction) and the verdict on the real data cannot be trusted.
      If C2 is close to C0, correction is not killing real signal, and a signal
      that disappears on the real data cannot be blamed on the correction.
""")
    say(f"all results written to {OUT_DIR}/")

def _r2_on(v, Xp):
    bb, *_ = np.linalg.lstsq(Xp, v, rcond=None)
    return float(1 - np.sum((v - Xp @ bb) ** 2) / max(np.sum((v - v.mean()) ** 2), 1e-12))

if __name__ == "__main__":
    main()
