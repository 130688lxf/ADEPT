#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 validate_f1_families.py -- do parental genomics predict F1 family means,
 computed the way the pipeline actually defines a cross?
================================================================================

 THREE CONVENTIONS THIS CHECK ENFORCES
 ------------------------------------------------------------------------------
 (1) NO MID-PARENT AVERAGING OF DOMINANCE. Mid-parent averaging is correct for
     the ADDITIVE value but wrong for dominance: a cross's expected dominance
     deviation is not the average of the parents' own dominance deviations, it
     is E[Hd] for that specific pairing. GeneticModel.expected_codes(Gp, Gq)
     computes this and is what the allocation stage uses, so the F1 check and
     the selection engine define "offspring value" identically. This matters
     most for phe4, the one dominance-positive trait, where averaging would
     mismeasure exactly the quantity the claim rests on.

 (2) ONE ALLELE-FREQUENCY BASIS FOR DOMINANCE. Hd for the breeding panel is
     not rebuilt on the breeding panel's own frequencies while d_hat was
     estimated on the reference panel's. expected_codes uses the model's
     stored reference mu / p_het / beta_d throughout, so the basis is
     consistent by construction.

 (3) a_hat IS VERIFIED BEFORE USE. The script asserts that
     cor(Ha_ref @ a_hat, reference additive kernel BLUP) is high before
     trusting any downstream number, and stops if it is not.

 HOW TO READ THE RESULT -- SET BEFORE LOOKING
 ------------------------------------------------------------------------------
 n = 14 families from only 4 sires. Effective information is far below 14
 because most predictor variance is a few sire levels (~3 df). Low power BY
 DESIGN. Reported with a permutation p, a Fisher 95% CI, and a leave-one-out
 range. A near-zero / non-significant r is the EXPECTED outcome of this design,
 NOT evidence the pipeline fails. Whatever the sign, it is reported as run.

 Run every requested trait; do not select traits after seeing results.

    python validate_f1_families.py \
        --ref_geno ref_all_phased.vcf --ref_pheno trait_imputed.csv \
        --breed_geno breed_all_phased.vcf --f1_csv f1_data_fixed.csv \
        --traits phe4 phe15 --out f1_validation
================================================================================
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mussel_pipeline as MP
import pig_epik_pipeline as P
import pig_epistasis_pipeline as Q
import pig_multigen_figures as MG


def wcorr(x, y, w):
    xm = np.average(x, weights=w)
    ym = np.average(y, weights=w)
    cov = np.average((x - xm) * (y - ym), weights=w)
    vx = np.average((x - xm) ** 2, weights=w)
    vy = np.average((y - ym) ** 2, weights=w)
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


def fisher_ci(r, n, alpha=0.05):
    if not np.isfinite(r) or n < 4 or abs(r) >= 1:
        return (np.nan, np.nan)
    from scipy.stats import norm
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    lo = z - norm.ppf(1 - alpha / 2) * se
    hi = z + norm.ppf(1 - alpha / 2) * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


def perm_p(pred, obs, w, r_obs, n_perm=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(obs))
    count = 0
    for _ in range(n_perm):
        rng.shuffle(idx)
        r = wcorr(pred[idx], obs, w)
        if np.isfinite(r) and abs(r) >= abs(r_obs) - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_geno", default="ref_all_phased.vcf")
    ap.add_argument("--ref_pheno", default="trait_imputed.csv")
    ap.add_argument("--breed_geno", default="breed_all_phased.vcf")
    ap.add_argument("--f1_csv", default="f1_data_fixed.csv")
    ap.add_argument("--traits", nargs="+", default=["phe4", "phe15"])
    ap.add_argument("--min_family_n", type=int, default=1)
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--a_hat_cor_min", type=float, default=0.99)
    ap.add_argument("--out", default="f1_validation")
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ids_r, chrom, pos, snp, Hr = MP.read_phased_vcf(a.ref_geno)
    Gr = MP.hap_to_geno(Hr)
    phe = pd.read_csv(a.ref_pheno)
    ia, ib = MP.align_ids(ids_r, phe.iloc[:, 0].values)
    ids_r, Gr, phe = ids_r[ia], Gr[ia], phe.iloc[ib].reset_index(drop=True)

    ids_b, chrom_b, pos_b, snp_b, Hb = MP.read_phased_vcf(a.breed_geno)
    Gb = MP.hap_to_geno(Hb)
    if not np.array_equal(snp, snp_b):
        raise SystemExit("marker sets differ between reference and breeding "
                         "VCFs; cross-panel scoring would be meaningless")
    id_b_index = {str(x): i for i, x in enumerate(ids_b.astype(str))}

    f1 = pd.read_csv(a.f1_csv)
    for col in ("group", "M", "F", "age_days"):
        if col not in f1.columns:
            raise SystemExit(f"F1 file missing column '{col}'")
    for t in a.traits:
        if t not in f1.columns:
            raise SystemExit(f"F1 file missing trait column '{t}'")
        if not np.issubdtype(f1[t].dtype, np.number):
            raise SystemExit(f"F1 trait column '{t}' is not numeric")

    missing = sorted((set(f1["M"]) | set(f1["F"])) - set(id_b_index))
    if missing:
        raise SystemExit(
            f"these F1 parents are not in the breeding panel: {missing}\n"
            "  resolve the id mapping rather than dropping families silently")

    n_sires = f1["M"].nunique()
    print(f"F1: {len(f1)} offspring in {f1['group'].nunique()} families, "
          f"from {n_sires} sires and {f1['F'].nunique()} dams")
    print(f"age spread: {f1['age_days'].min():.0f}-{f1['age_days'].max():.0f} "
          f"days")
    if n_sires <= 5:
        print(f"  POWER WARNING: only {n_sires} sires -- most predictor "
              f"variance is a few sire levels; effective df << family count.")

    per_trait_r = []
    summary = []
    for trait in a.traits:
        print(f"\n{'='*70}\n{trait}\n{'='*70}")
        y = phe[trait].values.astype(float)
        keep = np.isfinite(y)
        Gk = Gr[keep]
        Ha_r, Hd_r = P.noia(Gk)
        qc = Q.CFG(); qc.CACHE_DIR = "f1_ref_cache"; qc.DO_CIS = False
        K, meta = Q.build_kernels(qc, Gk, Ha_r, Hd_r, chrom, keep_xc=False)
        Ksub = {k: K[k] for k in ("A", "D") if k in K}
        yk = y[keep]
        gm = MG.GeneticModel(Gk, Ha_r, Hd_r, chrom, Ksub, meta, yk,
                             np.ones((len(yk), 1)))

        blup_A = gm.var.get("A", 0.0) * (K["A"] @ gm.alpha)
        pred_A = Ha_r @ gm.a_hat
        c = (float(np.corrcoef(pred_A, blup_A)[0, 1])
             if np.std(blup_A) > 1e-12 else np.nan)
        print(f"  a_hat check: cor(Ha@a_hat, additive kernel BLUP) = {c:.4f}")
        if not (np.isfinite(c) and c >= a.a_hat_cor_min):
            raise SystemExit(
                f"  a_hat does not reproduce the additive kernel BLUP "
                f"(cor={c:.4f} < {a.a_hat_cor_min}); predictions aborted.")

        rows = []
        for grp, g in f1.groupby("group"):
            sire, dam = str(g["M"].iloc[0]), str(g["F"].iloc[0])
            Gp = Gb[id_b_index[sire]][None, :]
            Gq = Gb[id_b_index[dam]][None, :]
            e_ha, e_hd = gm.expected_codes(Gp, Gq)
            pred = float((e_ha @ gm.a_hat)[0] + (e_hd @ gm.d_hat)[0])
            if gm.use_epi:
                pred += float(gm.epi_value(e_ha)[0])
            rows.append(dict(family=grp, sire=sire, dam=dam,
                             n_offspring=len(g), pred_cross_value=pred,
                             obs_mean=g[trait].mean(),
                             mean_age=g["age_days"].mean()))
        F = pd.DataFrame(rows)
        F = F[F.n_offspring >= a.min_family_n].reset_index(drop=True)
        w = F.n_offspring.values.astype(float)

        sd_pred = float(np.std(F.pred_cross_value))
        sd_obs = float(np.std(F.obs_mean))
        scale_flag = sd_pred > sd_obs

        r_raw = wcorr(F.pred_cross_value.values, F.obs_mean.values, w)

        def resid_on_age(v):
            A = np.column_stack([np.ones(len(F)), F.mean_age.values])
            beta, *_ = np.linalg.lstsq(A * np.sqrt(w[:, None]),
                                       v * np.sqrt(w), rcond=None)
            return v - A @ beta
        r_age = wcorr(resid_on_age(F.obs_mean.values),
                      resid_on_age(F.pred_cross_value.values), w)

        lo, hi = fisher_ci(r_raw, len(F))
        p_perm = perm_p(F.pred_cross_value.values, F.obs_mean.values, w,
                        r_raw, a.n_perm, a.seed)
        loo = []
        for i in range(len(F)):
            m = np.ones(len(F), bool); m[i] = False
            loo.append(wcorr(F.pred_cross_value.values[m],
                             F.obs_mean.values[m], w[m]))
        loo = np.array(loo)

        F.to_csv(os.path.join(a.out, f"{trait}_family_means.csv"), index=False)
        print(F.round(4).to_string(index=False))
        print(f"\n  weighted r (predicted cross value vs observed family mean):")
        print(f"    raw            r = {r_raw:+.3f}   (n = {len(F)} families)")
        print(f"    age-corrected  r = {r_age:+.3f}")
        print(f"    Fisher 95% CI    [{lo:+.3f}, {hi:+.3f}]")
        print(f"    permutation p    {p_perm:.3f}  ({a.n_perm} shuffles)")
        print(f"    leave-one-out r  {loo.min():+.3f} to {loo.max():+.3f}")
        print(f"    pred sd {sd_pred:.4f} vs obs sd {sd_obs:.4f}"
              + ("   [FLAG: predicted sd exceeds observed -- scale inflated]"
                 if scale_flag else "   [ok: predicted sd <= observed]"))
        if p_perm > 0.05:
            print(f"    -> not significant at n={len(F)}; consistent with an "
                  f"underpowered {n_sires}-sire design, NOT evidence of "
                  f"pipeline failure")
        per_trait_r.append(r_raw)
        summary.append(dict(trait=trait, n_families=len(F), r_raw=float(r_raw),
                            r_age=float(r_age), ci_lo=lo, ci_hi=hi,
                            perm_p=float(p_perm), loo_min=float(loo.min()),
                            loo_max=float(loo.max()), sd_pred=sd_pred,
                            sd_obs=sd_obs, scale_inflated=bool(scale_flag)))

    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(a.out, "f1_summary.csv"), index=False)
    mean_r = float(np.nanmean(per_trait_r))
    print(f"\n{'='*70}\nCROSS-TRAIT AGGREGATE ({len(a.traits)} pre-declared "
          f"traits)\n{'='*70}")
    print(f"  mean weighted r across traits = {mean_r:+.3f}")
    print("  per trait: " + ", ".join(f"{t}={r:+.3f}"
                                      for t, r in zip(a.traits, per_trait_r)))
    print(f"\n  Reminder: n=14 families from {n_sires} sires. Consistency "
          f"check, not a validation; tests only whether higher-merit parents "
          f"produced higher-performing offspring families. No within-family "
          f"(Mendelian-sampling) test is possible -- no offspring genotypes.")
    print(f"\n-> {a.out}/f1_summary.csv")


if __name__ == "__main__":
    main()
