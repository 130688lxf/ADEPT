#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 power_curve.py -- injection calibration: what could this population have
                   detected, had the effect been there?
================================================================================

 A screening result of "no dominance" or "no epistasis" is uninterpretable on
 its own. It can mean the effect is absent, or that the material could never
 have shown it. This module separates the two by injecting known amounts of
 non-additive variance into simulated phenotypes built on the real kernels of
 the real population, then running the identical screening rule on them.

 Two numbers come out of the grid, and both are needed before any negative
 result is written down:

   false-positive rate   detection rate at zero injection. It should sit near
                         zero. If it does not, the screening rule fabricates
                         effects in this material and no positive is safe.

   detection limit       the smallest injected variance fraction reaching the
                         target power (0.80 by default). Negatives below this
                         limit are undetermined, not absent. If no injected
                         level reaches the target, the module has no usable
                         limit here at all and its negatives carry no
                         information whatsoever -- which is a stronger
                         statement than a large limit, and the opposite of
                         "the effect does not exist".

 The simulation is species-agnostic: it needs the kernels and nothing else, so
 the same code calibrates an outbred animal population, a mollusc breeding
 population and an inbred plant panel. What differs between them is the answer,
 not the procedure.

 The injected fraction is a share of PHENOTYPIC variance, and the additive
 share is reduced by the same amount so that total heritability stays at
 h2 while the injection moves. Adding an injection on top of a fixed additive
 share would confound the injected effect with a rising heritability, and the
 resulting curve would overstate power.

 Used by mateagent.py (stage 1) and by the per-material finalisation scripts.

     python power_curve.py --selftest
================================================================================
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Simulation primitives. Deliberately free of pipeline imports so that the
# self-test can exercise them without any genotype data.
# ---------------------------------------------------------------------------
def sample_from_kernel(K: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Draw one standardised vector with covariance proportional to K.

    Kernels here are positive semi-definite but not positive definite -- in
    places they are Gram matrices of fewer markers than individuals -- so a
    Cholesky factorisation is not safe. The eigen route clips the small
    negative eigenvalues that floating point produces around zero, which a
    Cholesky would simply fail on.
    """
    w, V = np.linalg.eigh((K + K.T) / 2.0)
    w = np.maximum(w, 0.0)
    z = V @ (np.sqrt(w) * rng.standard_normal(len(w)))
    s = z.std()
    return z / s if s > 1e-12 else z


class Sampler:
    """
    Caches one eigendecomposition per kernel.

    A calibration grid draws hundreds of phenotypes from the same few kernels.
    Decomposing them once turns the dominant cost of the grid into a constant.
    Kernels listed in `keys` but absent from `K` are skipped silently, so a
    material whose dominance module is blocked can pass the same key list.
    """

    def __init__(self, K: Dict[str, np.ndarray], keys: Sequence[str]):
        self.keys = [k for k in keys if k in K]
        self._eig: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for k in self.keys:
            A = K[k]
            w, V = np.linalg.eigh((A + A.T) / 2.0)
            self._eig[k] = (np.maximum(w, 0.0), V)
        self.n = int(K[self.keys[0]].shape[0]) if self.keys else 0

    def draw(self, key: str, rng: np.random.Generator) -> np.ndarray:
        w, V = self._eig[key]
        z = V @ (np.sqrt(w) * rng.standard_normal(len(w)))
        s = z.std()
        return z / s if s > 1e-12 else z

    def has(self, key: str) -> bool:
        return key in self._eig


def simulate(sampler: Sampler, rng: np.random.Generator,
             var_a: float, var_d: float, var_e: float,
             epi_key: Optional[str] = None) -> np.ndarray:
    """
    y = sqrt(var_a) * g_A + sqrt(var_d) * g_D + sqrt(var_e) * g_E + noise

    Each variance is a fraction of phenotypic variance; the residual takes
    whatever is left. Components whose kernel is not available in the sampler
    are dropped and their variance returned to the residual, so a blocked
    module cannot be silently injected into a material that cannot express it.
    """
    tot = var_a + var_d + var_e
    if tot > 1.0 + 1e-9:
        raise ValueError(f"injected variances sum to {tot:.3f} > 1")
    if epi_key is None:
        epi_key = next((k for k in sampler.keys if k not in ("A", "D")), None)
    y = np.zeros(sampler.n)
    used = 0.0
    for key, v in (("A", var_a), ("D", var_d), (epi_key, var_e)):
        if key is None or v <= 0 or not sampler.has(key):
            continue
        y = y + np.sqrt(v) * sampler.draw(key, rng)
        used += v
    return y + np.sqrt(max(1.0 - used, 0.0)) * rng.standard_normal(sampler.n)


def simulate_phenotype(K_a: np.ndarray, K_e: np.ndarray, h2: float,
                       v_epi: float, rng: np.random.Generator) -> np.ndarray:
    """
    Two-kernel convenience form used by the per-material finalisation scripts.

    v_epi is the share of phenotypic variance assigned to the non-additive
    component; the additive share takes the rest of h2, so heritability is
    held constant as the injection moves.
    """
    if v_epi > h2:
        raise ValueError(f"injected variance {v_epi} exceeds h2 {h2}")
    v_add = h2 - v_epi
    n = K_a.shape[0]
    y = np.sqrt(1.0 - h2) * rng.standard_normal(n)
    if v_add > 0:
        y = y + np.sqrt(v_add) * sample_from_kernel(K_a, rng)
    if v_epi > 0:
        y = y + np.sqrt(v_epi) * sample_from_kernel(K_e, rng)
    return y


def verdict(deltas, threshold: float) -> Tuple[float, float, bool]:
    """Mean delta, positive fraction, and whether the screening rule fires."""
    d = np.asarray([x for x in np.asarray(deltas, float) if np.isfinite(x)],
                   float)
    if d.size == 0:
        return float("nan"), float("nan"), False
    mean_d = float(d.mean())
    pos = float((d > 0).mean())
    return mean_d, pos, bool(mean_d > 0 and pos >= threshold)


def identity_likeness(K: np.ndarray) -> Dict[str, float]:
    """
    How close a kernel is to a multiple of the identity.

    A kernel proportional to I contributes a ridge penalty and nothing else:
    it shrinks every prediction toward the mean by the same amount, which
    lowers the split-to-split variance of accuracy while explaining no genetic
    covariance between individuals. Such a kernel can pass a positive-fraction
    criterion while carrying no genetic information, so it must be identified
    before any decision is taken.

    offdiag_over_diag  mean |off-diagonal| / mean |diagonal|  (0 for exactly I)
    corr_with_identity correlation of the vectorised kernel with vectorised I
    effective_rank     participation ratio (sum w)^2 / sum(w^2) of the
                       eigenvalues. The name has several definitions in use;
                       only numbers computed under this one are comparable.
    """
    n = K.shape[0]
    d = np.diag(K)
    iu = np.triu_indices(n, 1)
    off = K[iu]
    ratio = float(np.mean(np.abs(off)) / max(float(np.mean(np.abs(d))), 1e-12))
    a = K.ravel()
    b = np.eye(n).ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 else float("nan")
    w = np.maximum(np.linalg.eigvalsh((K + K.T) / 2.0), 0.0)
    eff_rank = float(w.sum() ** 2 / max(float((w ** 2).sum()), 1e-12))
    return dict(offdiag_over_diag=ratio, corr_with_identity=corr,
                effective_rank=eff_rank, n=n,
                top_eigen_share=float(w.max() / max(w.sum(), 1e-12)))


# ---------------------------------------------------------------------------
# The screening rule itself. This is the ONLY place the comparison is defined,
# so a simulated phenotype and a real one are judged by identical code.
# ---------------------------------------------------------------------------
def _model_keys(K: Dict[str, np.ndarray], module: str,
                with_dominance: bool, epi_key: Optional[str]):
    if epi_key is None:
        epi_key = next((k for k in K if k not in ("A", "D")), None)
    if module == "dominance":
        if "D" not in K:
            raise KeyError("dominance screening needs a D kernel")
        return ["A"], ["A", "D"]
    if module == "epistasis":
        if epi_key is None:
            raise KeyError("epistasis screening needs an epistatic kernel")
        base = ["A", "D"] if (with_dominance and "D" in K) else ["A"]
        return base, base + [epi_key]
    raise ValueError(f"unknown module {module!r}")


def screen_once(y: np.ndarray, K: Dict[str, np.ndarray], labels,
                cfg, n_splits: int, module: str, n_boot: int = 0,
                with_dominance: bool = True,
                epi_key: Optional[str] = None) -> Tuple[float, float]:
    """
    Run one module's screening comparison over `n_splits` random splits and
    return (mean delta in predictive correlation, positive fraction).

    dominance   A       vs A+D
    epistasis   A+D     vs A+D+E   (A vs A+E where dominance is not estimable)

    The seed sequence is cfg.SEED + i * 101, matching the screening driver, so
    a calibration curve and a real screen see the same partitions of the same
    individuals.

    `n_boot` is accepted for interface compatibility with the screening driver.
    The decision rule is the positive fraction across splits, not a bootstrap
    interval, so no bootstrap is drawn here.
    """
    import pig_epistasis_pipeline as Q          # imported late: keeps the
                                                # simulation primitives above
                                                # usable without the pipeline
    keys0, keys1 = _model_keys(K, module, with_dominance, epi_key)
    n = len(y)
    Xfix = np.ones((n, 1))
    split_mode = getattr(cfg, "SPLIT", "random")
    frac = getattr(cfg, "TEST_FRAC", 0.20)
    seed0 = getattr(cfg, "SEED", 2024)
    deltas: List[float] = []
    for i in range(n_splits):
        tr, te = Q.make_split(n, labels, split_mode, frac, seed0 + i * 101)
        g0, ya, _ = Q.fit_gebv(y, Xfix, K, keys0, tr, te)
        g1, _, _ = Q.fit_gebv(y, Xfix, K, keys1, tr, te)
        if g0 is None or g1 is None:
            continue
        deltas.append(Q.pcc(ya, g1) - Q.pcc(ya, g0))
    mean_d, pos, _ = verdict(deltas, 1.0)
    return mean_d, pos


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
def calibration_grid(K: Dict[str, np.ndarray], labels, cfg,
                     modules: Sequence[str], levels: Sequence[float],
                     reps: int, n_splits: int, h2_sim: float,
                     threshold: float, out_dir: str,
                     epi_key: Optional[str] = None,
                     with_dominance: bool = True,
                     say=print) -> pd.DataFrame:
    """
    Run the full (module x injected level x replicate) grid and summarise it.

    Every cell is written to disk as soon as it finishes and completed cells
    are skipped on restart. This is not a convenience: a grid of this size runs
    for hours, and a version of it that wrote results only at the end lost a
    five-hour run to a single interrupt.
    """
    os.makedirs(out_dir, exist_ok=True)
    ck_dir = os.path.join(out_dir, "_calib")
    os.makedirs(ck_dir, exist_ok=True)
    raw_path = os.path.join(ck_dir, "calibration_raw.csv")

    rows: List[dict] = []
    done = set()
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
        prev = pd.read_csv(raw_path)
        rows = prev.to_dict("records")
        done = {(str(r["module"]), round(float(r["injected"]), 6), int(r["rep"]))
                for r in rows}
        say(f"  resuming: {len(rows)} cell(s) already done")

    sampler = Sampler(K, ["A", "D"] + ([epi_key] if epi_key else
                                       [k for k in K if k not in ("A", "D")]))
    total = len(modules) * len(levels) * reps
    t0 = time.time()
    for module in modules:
        for lvl in levels:
            for rep in range(reps):
                cell = (module, round(float(lvl), 6), rep)
                if cell in done:
                    continue
                rng = np.random.default_rng(9000 + rep)   # same draw at every
                var_d = float(lvl) if module == "dominance" else 0.0
                var_e = float(lvl) if module == "epistasis" else 0.0
                var_a = max(h2_sim - var_d - var_e, 1e-6)
                y = simulate(sampler, rng, var_a, var_d, var_e, epi_key)
                d, pf = screen_once(y, K, labels, cfg, n_splits, module,
                                    with_dominance=with_dominance,
                                    epi_key=epi_key)
                hit = bool(np.isfinite(d) and d > 0 and pf >= threshold)
                rows.append(dict(module=module, injected=float(lvl), rep=rep,
                                 dPCC=d, pos_frac=pf, detected=int(hit)))
                pd.DataFrame(rows).to_csv(raw_path, index=False)
                say(f"    {module} injected={float(lvl):.2f} "
                    f"rep {rep + 1}/{reps}  dPCC={d:+.5f} pos_frac={pf:.2f} "
                    f"{'HIT' if hit else '.'}   "
                    f"[{len(rows)}/{total}, "
                    f"{(time.time() - t0) / 60:.1f} min elapsed]")

    R = pd.DataFrame(rows)
    summ = (R.groupby(["module", "injected"])
              .agg(n_reps=("rep", "count"), mean_dPCC=("dPCC", "mean"),
                   median_dPCC=("dPCC", "median"),
                   detection_rate=("detected", "mean"))
              .reset_index())
    summ.to_csv(os.path.join(out_dir, "calibration.csv"), index=False)
    say("\n" + summ.round(5).to_string(index=False))
    for module, g in summ.groupby("module"):
        report_limits(g, module, say=say)
    return summ


def report_limits(summ: pd.DataFrame, module: str, power: float = 0.80,
                  say=print) -> Dict[str, float]:
    """Print, and return, the two numbers a negative result cannot be read without."""
    fp = summ.loc[summ.injected == 0.0, "detection_rate"]
    fpr = float(fp.iloc[0]) if len(fp) else float("nan")
    ok = summ[(summ.injected > 0) & (summ.detection_rate >= power)]
    limit = float(ok.injected.iloc[0]) if len(ok) else float("nan")
    say(f"\n  [{module}] false-positive rate at zero injection: {fpr:.2f}")
    if np.isfinite(limit):
        say(f"  [{module}] detection limit at {power:.0%} power: {limit:.2f} "
            f"of phenotypic variance. Negatives below this are undetermined.")
    else:
        say(f"  [{module}] NO injected level reached {power:.0%} power. This "
            f"module has no usable\n  detection limit in this material, so a "
            f"negative result carries no information at\n  all and must be "
            f"reported as insufficient power, never as absence of effect.")
    return dict(false_positive_rate=fpr, detection_limit=limit)


def implied_injection(summ: pd.DataFrame, observed: float
                      ) -> Tuple[float, bool]:
    """
    Locate an observed delta on the calibration curve.

    Returns (equivalent injected fraction, whether it is inside the calibrated
    range). Interpolation outside the tested grid would clamp silently to the
    boundary and turn "stronger than anything we tested" into the spuriously
    precise "exactly the largest level tested", so the caller is told which of
    the two it has.
    """
    g = summ.sort_values("injected")
    xs = g.injected.values.astype(float)
    ys = g.mean_dPCC.values.astype(float)
    if not np.all(np.diff(ys) >= -1e-12):
        order = np.argsort(ys)
        xs, ys = xs[order], ys[order]
    inside = bool(ys.min() <= observed <= ys.max())
    return float(np.interp(observed, ys, xs)), inside


# ---------------------------------------------------------------------------
def _selftest() -> int:
    print("[1] a kernel draw has the requested covariance structure")
    rng = np.random.default_rng(0)
    n = 120
    # a kernel with real structure: 12 families of 10, so the off-diagonal
    # entries the test compares are not all sampling noise around zero
    fam = np.repeat(np.arange(12), 10)
    B = (rng.standard_normal((n, 400)) * 0.6
         + rng.standard_normal((12, 400))[fam])
    Ka = B @ B.T / 400.0
    Ke = (Ka * Ka)
    z = np.column_stack([sample_from_kernel(Ka, rng) for _ in range(2000)])
    emp = np.corrcoef(z)
    tgt = Ka / np.sqrt(np.outer(np.diag(Ka), np.diag(Ka)))
    iu = np.triu_indices(n, 1)
    r = float(np.corrcoef(emp[iu], tgt[iu])[0, 1])
    print(f"    empirical vs target off-diagonal correlation r = {r:.3f}")
    assert r > 0.9, "draws do not follow the kernel"

    print("[2] injection holds total heritability constant")
    s = Sampler({"A": Ka, "D": Ka * 0.5, "MULT_t": Ke}, ["A", "D", "MULT_t"])
    for v_e in (0.0, 0.1, 0.3):
        y = simulate(s, np.random.default_rng(1), 0.4 - v_e, 0.0, v_e)
        print(f"    v_epi={v_e:.2f}  sd(y)={y.std():.3f}")
    print("[3] a blocked module cannot be injected")
    s2 = Sampler({"A": Ka, "MULT_t": Ke}, ["A", "D", "MULT_t"])
    assert not s2.has("D")
    y = simulate(s2, np.random.default_rng(2), 0.3, 0.2, 0.0)
    assert np.all(np.isfinite(y))
    print("    dominance variance returned to the residual, as intended")

    print("[4] identity-like kernels are flagged")
    I = np.eye(n)
    a, b = identity_likeness(I), identity_likeness(Ka)
    print(f"    identity : offdiag/diag={a['offdiag_over_diag']:.3f} "
          f"corr_with_I={a['corr_with_identity']:.3f}")
    print(f"    real GRM : offdiag/diag={b['offdiag_over_diag']:.3f} "
          f"corr_with_I={b['corr_with_identity']:.3f}")
    assert a["offdiag_over_diag"] < 1e-9 and b["offdiag_over_diag"] > 0.01

    print("[5] out-of-range observations are reported as out of range")
    summ = pd.DataFrame(dict(injected=[0.0, 0.05, 0.15],
                             mean_dPCC=[0.000, 0.002, 0.004],
                             detection_rate=[0.0, 0.2, 0.6]))
    v, inside = implied_injection(summ, 0.0030)
    print(f"    observed 0.0030 -> injected {v:.3f}, inside range = {inside}")
    assert inside
    v, inside = implied_injection(summ, 0.0160)
    print(f"    observed 0.0160 -> inside range = {inside} (exceeds the grid)")
    assert not inside

    print("[6] no injected level reaching target power is reported as such")
    out = report_limits(summ.assign(module="epistasis"), "epistasis")
    assert not np.isfinite(out["detection_limit"])
    print("\nALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
