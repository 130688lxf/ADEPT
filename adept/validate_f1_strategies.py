#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 validate_f1_strategies.py -- test the six allocation strategies against the
                              real F1 families
================================================================================

 Allocation reports six arms, so the F1 check has to compare all of them: for
 each of the 14 realised families, every strategy's prediction is computed and
 compared with the observed family mean.

 WHAT THIS DESIGN CAN AND CANNOT CLAIM -- stated first, to prevent
 over-reading. The F1 generation has phenotypes but no genotypes, and cannot be
 genotyped retrospectively at the sample level. Therefore:
   1. only FAMILY-level consistency is available (n=14), not individual-level
      predictive accuracy;
   2. within-family Mendelian sampling variance cannot be tested at all;
   3. a family mean averages over segregation, so this mainly probes ADDITIVE
      transmission and has very little power for dominance or epistasis;
   4. the 14 families descend from only a few sires, so the effective degrees
      of freedom are far below 14.
 This section is therefore a CONSISTENCY CHECK, not a validation, and the
 result is reported as it comes out either way.

 Two levels of comparison:
   A. Prediction level: the weighted correlation (weighted by the number of
      offspring) between the observed family means and the predictions of
      ADD / ADD+DOM / FULL / weighted. two_stage and add_floor are SELECTION
      RULES rather than predictors and do not appear here; they use the FULL
      predictions and would duplicate that row.
   B. Selection level: if a strategy were used to pick the top k of the 14
      families, how did the picked families actually perform. This is the
      direct analogue of the strategy, but n=14 makes it very weak and it is
      reported descriptively only.

 Implementation constraints, taken from the allocation engine rather than
 shortcut:
   - expected offspring codes come from expected_codes on the SPECIFIC parent
     pair, never from averaging mid-parent GEBVs (mid-parent averaging is valid
     for the additive value only; the mid-parent dominance deviation does not
     predict the offspring's)
   - the reference-population NOIA basis is used throughout; frequencies are
     never recomputed on the breeding population
   - before anything else, the script asserts that the marker effects
     reconstruct the reference additive kernel BLUP (cor >= 0.99) and stops if
     they do not
   - a scale diagnostic is reported: if the sd of the predicted family means
     exceeds the observed sd, that is a scale error, not good prediction
================================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pig_epik_pipeline as P
import pig_epistasis_pipeline as Q
from mussel_trait_pipeline import (read_phased_vcf, detect_id_col,
                                   match_ids_to_pheno, noia_fit, say)

try:
    from trait_names import pretty as _pretty
except ImportError:
    def _pretty(t, sp=None): return str(t)

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def wcorr(x, y, w):
    """Pearson correlation weighted by the number of offspring."""
    x, y, w = map(np.asarray, (x, y, w))
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    sx = np.sqrt((w * (x - mx) ** 2).sum())
    sy = np.sqrt((w * (y - my) ** 2).sum())
    return float(cov / (sx * sy)) if sx > 0 and sy > 0 else np.nan


def fisher_ci(r, n, alpha=0.05):
    if not np.isfinite(r) or n < 4 or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    q = stats.norm.ppf(1 - alpha / 2) if HAVE_SCIPY else 1.959964
    return (float(np.tanh(z - q * se)), float(np.tanh(z + q * se)))


def perm_p(x, y, w, n_perm=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(wcorr(x, y, w))
    if not np.isfinite(obs):
        return np.nan
    cnt = 0
    yv = np.asarray(y)
    for _ in range(n_perm):
        if abs(wcorr(x, rng.permutation(yv), w)) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def run_one(a, trait):
    """Run one trait and return its summary row."""
    class _A: pass
    z = _A()
    for k, v in vars(a).items(): setattr(z, k, v)
    z.trait = trait
    if a.pairs_dir:
        cand = os.path.join(a.pairs_dir, f'pairval_{trait}', 'usable_pairs.csv')
        z.pairs_csv = cand if os.path.exists(cand) else None
    z.out = os.path.join(a.out, trait)
    os.makedirs(z.out, exist_ok=True)
    return _run(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f1_csv", required=True,
                    help="one row per offspring; needs family, sire, dam and phenotype columns")
    ap.add_argument("--traits", nargs="+", required=True,
                    help="one or more traits, e.g. --traits phe1 phe2 phe4 ...")
    ap.add_argument("--pairs_dir", default=None,
                    help="directory containing pairval_<trait>/usable_pairs.csv; when given, "
                         "the usable-pair file is picked per trait automatically")
    ap.add_argument("--ref_vcf", default="ref_all_phased.vcf")
    ap.add_argument("--ref_pheno", default="trait_imputed.csv")
    ap.add_argument("--breed_vcf", default="breed_all_phased.vcf")
    ap.add_argument("--pairs_csv", default=None,
                    help="usable_pairs.csv; without it only ADD and ADD+DOM are compared")
    ap.add_argument("--pair_type", default="MULT")
    ap.add_argument("--family_col", default=None)
    ap.add_argument("--sire_col", default=None)
    ap.add_argument("--dam_col", default=None)
    ap.add_argument("--age_col", default=None)
    ap.add_argument("--pheno_col", default=None,
                    help="phenotype column in the F1 table; defaults to --trait")
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--out", default="f1_strategies")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    allrows = []
    for t in a.traits:
        say("\n" + "#" * 78)
        say(f"#  {t}  ({_pretty(t, 'mussel')})")
        say("#" * 78)
        try:
            r = run_one(a, t)
            if r is not None:
                allrows.append(r)
        except SystemExit as e:
            say(f"  [skipped {t}] {e}")
        except Exception as e:
            say(f"  [skipped {t}] {type(e).__name__}: {e}")

    if allrows:
        ALL = pd.concat(allrows, ignore_index=True)
        p = os.path.join(a.out, "F1_all_traits_summary.csv")
        ALL.to_csv(p, index=False)
        say("\n" + "=" * 78)
        say("Summary across traits (prediction level, correlations weighted by offspring count)")
        say("=" * 78)
        piv = ALL[ALL.phenotype == "raw"].pivot(
            index="trait_label", columns="strategy", values="r")
        say(piv.round(3).to_string())
        say(f"\n-> {p}")
    else:
        say("\nNo trait completed successfully.")


def _run(a):
    TR = _pretty(a.trait, "mussel")

    # ---------------- read the F1 table ----------------
    F = pd.read_csv(a.f1_csv)
    say(f"[F1] {a.f1_csv}: {len(F)} rows, columns {list(F.columns)}")

    def guess(col, exact, substr):
        """Exact match first, substring match second, and the two must stay
        separate: 'female' contains 'male' and 'num' contains 'm', so mixing them
        would read the dam column, or an id column, as the sire."""
        if col:
            return col
        lower = {str(k).strip().lower(): k for k in F.columns}
        for c in exact:
            if c in lower:
                return lower[c]
        for k in F.columns:
            kl = str(k).strip().lower()
            if any(c in kl for c in substr):
                return k
        return None

    fam = guess(a.family_col,
                ["group", "family", "fam", "grp",
                 "\u5bb6\u7cfb", "\u7ec4\u5408", "\u7ec4"],
                ["family", "\u5bb6\u7cfb", "\u7ec4\u5408"])
    sire = guess(a.sire_col,
                 ["m", "sire", "father", "male", "sire_id",
                  "\u7236", "\u516c", "\u516c\u672c", "\u7236\u672c"],
                 ["sire", "father", "\u516c\u672c", "\u7236\u672c"])
    dam = guess(a.dam_col,
                ["f", "dam", "mother", "female", "dam_id",
                 "\u6bcd", "\u6bcd\u672c"],
                ["dam", "mother", "\u6bcd\u672c"])
    phe = a.pheno_col or (a.trait if a.trait in F.columns else None)
    age = guess(a.age_col,
                ["age_days", "age", "days", "\u65e5\u9f84"],
                ["age", "\u65e5\u9f84"])
    if sire is not None and dam is not None and sire == dam:
        raise SystemExit(f"sire and dam were resolved to the same column ({sire}); "
                         f"specify --sire_col / --dam_col explicitly")
    miss = [n for n, v in [("family", fam), ("sire", sire), ("dam", dam),
                           ("phenotype", phe)] if v is None]
    if miss:
        raise SystemExit(
            f"could not identify these columns in the F1 table: {miss}\n"
            f"columns present: {list(F.columns)}\n"
            f"specify them with --family_col / --sire_col / --dam_col / --pheno_col")
    say(f"  family={fam}  sire={sire}  dam={dam}  phenotype={phe}  "
        f"age={age or 'none'}")

    F = F.dropna(subset=[fam, sire, dam, phe])
    g = F.groupby([fam, sire, dam], sort=False)
    FAM = g[phe].agg(["mean", "size"]).reset_index()
    FAM.columns = [fam, sire, dam, "obs_mean", "n_off"]
    if age:
        FAM["age_mean"] = g[age].mean().values
    say(f"  {len(FAM)} families, {int(FAM.n_off.sum())} offspring, "
        f"{FAM[sire].nunique()} sire(s), {FAM[dam].nunique()} dam(s)")
    if FAM[sire].nunique() < 5:
        say(f"  only {FAM[sire].nunique()} sire(s): the effective degrees of freedom are "
            f"far below the number of families, so this can only be a consistency check")

    # ---------------- estimate effects in the reference population ----------------
    say(f"\n[reference] {a.ref_vcf}")
    Xr, chrom, pos, snp_names, ref_ids = read_phased_vcf(a.ref_vcf)
    pheno = pd.read_csv(a.ref_pheno)
    pheno = pheno.set_index(detect_id_col(pheno))
    keep, matched, _ = match_ids_to_pheno(ref_ids, pheno, label="reference population")
    if not keep.all():
        Xr = Xr[keep]; ref_ids = [r for r, k in zip(ref_ids, keep) if k]
    y = pheno.loc[matched, a.trait].values.astype(float)

    Ha, Hd, mu, pAa, beta_d = noia_fit(Xr)
    Xx = (Xr > 0).astype(np.float64) * 2 - 1
    xc = Xx - Xx.mean(0); sd = xc.std(0); sd[sd < 1e-12] = 1.0
    Xc = xc / sd; del Xx, xc
    qc = Q.CFG(); qc.CACHE_DIR = os.path.join(a.out, "_kcache"); qc.DO_CIS = False
    K_ref, meta = Q.build_kernels(qc, Xr, Ha, Hd, chrom, keep_xc=False)
    name_to_idx = {s: k for k, s in enumerate(snp_names)}

    sel = []
    if a.pairs_csv and os.path.exists(a.pairs_csv) and os.path.getsize(a.pairs_csv):
        up = pd.read_csv(a.pairs_csv)
        if "trait" in up.columns:
            up = up[up.trait == a.trait]
        for r in up.itertuples():
            if r.snp_i in name_to_idx and r.snp_j in name_to_idx:
                sel.append((1.0, name_to_idx[r.snp_i], name_to_idx[r.snp_j],
                            a.pair_type))
    has_epi = len(sel) > 0
    say(f"  validated usable pairs: {len(sel)}")

    Xfix = np.ones((Xr.shape[0], 1))
    if has_epi:
        KT = Q.pairs_kernel(sel, a.pair_type, Xr, Ha, Hd, Xc)
        K = {k: K_ref[k] for k in ("A", "D")}; K["TOP"] = KT
        keys = ["A", "D", "TOP"]
    else:
        K = {k: K_ref[k] for k in ("A", "D")}; keys = ["A", "D"]
    mm, dual, var = Q.offspring_components(qc, Xr, Ha, Hd, chrom, K, meta,
                                           y, Xfix, keys)
    tot = sum(v for k, v in var.items() if k != "_resid") + var["_resid"]
    vf = {k: var[k] / tot for k in keys}
    say("  variance fractions: " + "  ".join(f"{k}={vf[k]:.4f}" for k in keys))

    a_hat = var["A"] * (Ha.T @ dual) / (meta["sA"] * (
        2 * np.sum((mu / 2) * (1 - mu / 2)) + 1e-12))
    d_hat = (var["D"] * (Hd.T @ dual) / (meta["sD"] * (
        np.sum(np.var(Hd, 0)) + 1e-12)) if "D" in var else np.zeros(Xr.shape[1]))

    # ---- correctness assertion: the marker effects must reconstruct the kernel BLUP ----
    recon = Ha @ a_hat
    kb = var["A"] * (K_ref["A"] @ dual)
    c = float(np.corrcoef(recon, kb)[0, 1])
    say(f"  [assertion] marker effects reconstruct the additive kernel BLUP: cor = {c:.4f}")
    if c < 0.99:
        raise SystemExit(f"reconstruction correlation {c:.4f} < 0.99: the marker effects "
                         f"are not a faithful decomposition of the kernel model, so every "
                         f"number downstream would be unreliable. Aborting.")

    if has_epi:
        pair_idx = [(i, j) for _, i, j, _ in sel]
        fm = np.array([(Ha[:, i] * Ha[:, j]).mean() for i, j in pair_idx])
        fs = np.array([(Ha[:, i] * Ha[:, j]).std() for i, j in pair_idx])
        fs[fs < 1e-10] = 1.0
        Z = np.column_stack([Q.pair_feature(a.pair_type, a.pair_type, Xr, Ha,
                                            Hd, Xc, i, j) for i, j in pair_idx])
        top_eff = var["TOP"] * (Z.T @ dual) / (Z.shape[1] + 1e-12)
        ai = np.array([p[0] for p in pair_idx]); aj = np.array([p[1] for p in pair_idx])

    # ---------------- breeding-population genotypes (the F1 parents) ----------------
    say(f"\n[breeding population] {a.breed_vcf}")
    Xb, cb, pb, nb, breed_ids = read_phased_vcf(a.breed_vcf)
    assert list(snp_names) == list(nb), "reference and breeding marker sets differ"
    bidx = {s: k for k, s in enumerate(breed_ids)}

    # align parent ids with the breeding VCF, tolerating the doubled c1_c1 form
    def find_parent(pid):
        s = str(pid).strip()
        for cand in (s, f"{s}_{s}", s.split("_")[0]):
            if cand in bidx:
                return bidx[cand]
        return None

    rows, unmatched = [], []
    for r in FAM.itertuples(index=False):
        si, di = find_parent(getattr(r, sire)), find_parent(getattr(r, dam))
        if si is None or di is None:
            unmatched.append((getattr(r, sire), getattr(r, dam)))
            continue
        rows.append((r, si, di))
    if unmatched:
        say(f"  [warning] the parents of {len(unmatched)} family/families were not found "
            f"in the breeding VCF, for example {unmatched[:3]}")
    if not rows:
        raise SystemExit("no family had both parents matched in the breeding VCF; check the identifier format")
    say(f"  both parents matched for {len(rows)}/{len(FAM)} family/families")

    # ---------------- expected offspring values, from the allocation engine ----------------
    recs = []
    for r, si, di in rows:
        pa, pb_ = Xb[si] / 2.0, Xb[di] / 2.0
        e_ha = (pa + pb_) - mu
        p_het = pa * (1 - pb_) + (1 - pa) * pb_
        e_hd = p_het - pAa - beta_d * e_ha
        gv_add = float(e_ha @ a_hat)
        gv_dom = float(e_hd @ d_hat)
        gv_epi = 0.0
        if has_epi:
            feat = (e_ha[ai] * e_ha[aj] - fm) / fs
            gv_epi = float(feat @ top_eff)
        rec = dict(family=getattr(r, fam), sire=getattr(r, sire),
                   dam=getattr(r, dam), n_off=int(r.n_off),
                   obs_mean=float(r.obs_mean),
                   gv_add=gv_add, gv_dom=gv_dom, gv_epi=gv_epi)
        if age:
            rec["age_mean"] = float(r.age_mean)
        recs.append(rec)
    D = pd.DataFrame(recs)

    # ---------------- predictions per strategy ----------------
    w_d = vf.get("D", 0) / max(vf.get("A", 1e-9), 1e-9)
    w_e = vf.get("TOP", 0) / max(vf.get("A", 1e-9), 1e-9)
    D["pred_ADD"] = D.gv_add
    D["pred_ADD_DOM"] = D.gv_add + D.gv_dom
    D["pred_FULL"] = D.gv_add + D.gv_dom + D.gv_epi
    D["pred_weighted"] = D.gv_add + w_d * D.gv_dom + w_e * D.gv_epi
    preds = ["pred_ADD", "pred_ADD_DOM", "pred_FULL", "pred_weighted"]
    say(f"\n  weighted arm: w_D={w_d:.4f}  w_E={w_e:.4f}")

    # age-adjusted variant
    obs = D.obs_mean.values.astype(float)
    w = D.n_off.values.astype(float)
    obs_adj = obs.copy()
    if age and D.age_mean.std() > 0:
        A_ = np.column_stack([np.ones(len(D)), D.age_mean.values])
        obs_adj = obs - A_ @ np.linalg.lstsq(A_, obs, rcond=None)[0] + obs.mean()

    # ---------------- A. prediction level ----------------
    out = []
    for p_ in preds:
        x = D[p_].values.astype(float)
        for tag, yv in [("raw", obs), ("age_adj", obs_adj)]:
            r = wcorr(x, yv, w)
            lo, hi = fisher_ci(r, len(D))
            pp = perm_p(x, yv, w, a.n_perm, seed=0) if HAVE_SCIPY or True else np.nan
            # leave one family out
            loo = [wcorr(np.delete(x, i), np.delete(yv, i), np.delete(w, i))
                   for i in range(len(D))]
            out.append(dict(strategy=p_.replace("pred_", ""), phenotype=tag,
                            n_families=len(D), r=round(r, 4),
                            ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                            perm_p=round(pp, 4),
                            loo_min=round(float(np.nanmin(loo)), 4),
                            loo_max=round(float(np.nanmax(loo)), 4)))
    R = pd.DataFrame(out)

    # scale diagnostic
    scale = {p_: dict(pred_sd=float(np.std(D[p_])), obs_sd=float(np.std(obs)),
                      ratio=float(np.std(D[p_]) / (np.std(obs) + 1e-12)))
             for p_ in preds}

    # ---------------- B. selection level ----------------
    selres = []
    for k in [3, 5]:
        if k >= len(D):
            continue
        best_actual = set(D.nlargest(k, "obs_mean").family)
        for p_ in preds:
            picked = set(D.nlargest(k, p_).family)
            selres.append(dict(top_k=k, strategy=p_.replace("pred_", ""),
                               hits=len(picked & best_actual),
                               mean_obs_of_picked=round(
                                   float(D[D.family.isin(picked)].obs_mean.mean()), 4),
                               mean_obs_overall=round(float(D.obs_mean.mean()), 4)))
    SEL = pd.DataFrame(selres)

    # ---------------- output ----------------
    D.to_csv(os.path.join(a.out, f"{TR}_family_predictions.csv"), index=False)
    R.to_csv(os.path.join(a.out, f"{TR}_strategy_correlations.csv"), index=False)
    if len(SEL):
        SEL.to_csv(os.path.join(a.out, f"{TR}_selection_check.csv"), index=False)
    json.dump(dict(trait=a.trait, trait_label=TR, n_families=len(D),
                   n_offspring=int(D.n_off.sum()), n_sires=int(D.sire.nunique()),
                   variance_fractions=vf, w_D=w_d, w_E=w_e,
                   n_usable_pairs=len(sel), recon_cor=c, scale_diag=scale),
              open(os.path.join(a.out, "run_info.json"), "w"),
              indent=2, default=str)

    say("\n" + "=" * 78)
    say(f"A. prediction level: predictions vs observed family means ({TR}, "
        f"n={len(D)} families, weighted by offspring count)")
    say("=" * 78)
    say(R.to_string(index=False))
    say("\nScale diagnostic (pred_sd / obs_sd; a value above 1 means the predicted\n"
        "variance exceeds the observed, which is a scale problem):")
    for p_, v in scale.items():
        flag = "  <- predicted variance exceeds observed; check the coding basis" if v["ratio"] > 1 else ""
        say(f"  {p_:<16} {v['ratio']:.3f}{flag}")
    if len(SEL):
        say("\n" + "=" * 78)
        say("B. selection level: picking the top k families with each strategy, how many\n"
            "   of the genuinely best families were picked")
        say("=" * 78)
        say(SEL.to_string(index=False))

    say("\n" + "=" * 78)
    say("How this must be read (state it in the manuscript, do not omit it)")
    say("=" * 78)
    say(f"  n={len(D)} families from {D.sire.nunique()} sire(s): the effective degrees of\n"
        f"  freedom are far below the number of families.")
    say("  This is a consistency check, not a validation. A positive or negative\n"
        "  correlation is not sufficient to rank the strategies.")
    say("  A family mean averages over segregation, so it mainly probes additive\n"
        "  transmission and has little power for dominance or epistasis.")
    R.insert(0, "trait", a.trait)
    R.insert(1, "trait_label", TR)
    say(f"\n-> {a.out}/")
    return R


if __name__ == "__main__":
    main()
