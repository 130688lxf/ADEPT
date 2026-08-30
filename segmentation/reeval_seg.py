#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 reeval_seg.py -- re-evaluate trained weights without retraining
================================================================================

 Three points of the EVALUATION protocol, none of them in training:
   1. data augmentation was left on for validation and test (augmenting the
      training set is correct; augmenting the evaluation set is not)
   2. the 5x5 closing plus largest connected component post-processing was
      missing, although the published manual protocol includes it
   3. illumination robustness pooled areas from different individuals into one
      CV, which mostly measures shell-size variation between individuals

 The trained weights themselves are unaffected, so best_iou.pth only needs a
 fresh forward pass. Each fold's validation set holds a few dozen images and a
 full pass over all seven variants usually takes on the order of ten minutes,
 against tens of hours to retrain.

 The splits reproduce the training scripts exactly: the same seed and randperm
 to separate trainval from test, and the same KFold(shuffle=True,
 random_state=seed) for the folds, so the same samples are evaluated.

 Writes <run_dir>/results_summary_reeval.csv and test_results_reeval.csv
 without overwriting the originals, so the two protocols can be compared.

     python reeval_seg.py --all
     python reeval_seg.py --script unetmamba.py --run_dir outputs_UNet_Mamba/run_1
================================================================================
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from scipy import ndimage as _ndi


def say(*a):
    print(*a, flush=True)


def load_script(path):
    """Import a training script as a module; its main() does not run thanks to the __name__ guard."""
    name = "seg_" + os.path.splitext(os.path.basename(path))[0].replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def find_model_class(mod, path):
    """Find the model class actually used, from `model = XXX(` inside main()."""
    src = open(path, newline="").read()
    m = re.search(r"model\s*=\s*(\w+)\s*\(", src)
    if m and hasattr(mod, m.group(1)):
        return getattr(mod, m.group(1)), m.group(1)
    cands = [(k, v) for k, v in vars(mod).items()
             if isinstance(v, type) and issubclass(v, nn.Module)
             and v.__module__ == mod.__name__
             and k.lower().startswith(("unet", "transunet"))]
    if len(cands) == 1:
        return cands[0][1], cands[0][0]
    raise SystemExit(f"could not determine the model class; candidates: "
                     f"{[k for k, _ in cands]}. "
                     f"Specify it with --model_class.")


def postprocess(pred):
    """5x5 closing plus the largest connected component, as in the published manual protocol."""
    out = pred.detach().cpu().numpy()
    st = np.ones((5, 5), dtype=bool)
    for b in range(out.shape[0]):
        m = out[b, 0] > 0.5
        if m.sum() == 0:
            continue
        m = _ndi.binary_closing(m, structure=st)
        lab, n = _ndi.label(m)
        if n > 1:
            sizes = _ndi.sum(m, lab, range(1, n + 1))
            m = (lab == (int(np.argmax(sizes)) + 1))
        out[b, 0] = m.astype(np.float32)
    return torch.from_numpy(out).to(pred.device)


def eval_loader(mod, model, loader, device, do_post):
    keys = ['iou', 'dice', 'hd95', 'asd', 'area_error_percent',
            'centroid_distance_px', 'boundary_f1']
    acc = {k: [] for k in keys}
    model.eval()
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            if do_post:
                # reuse calc_metrics from the training script, but first turn the
                # post-processed mask back into an equivalent logit tensor
                pred = (torch.sigmoid(logits) > 0.5).float()
                pred = postprocess(pred)
                logits_eq = (pred * 2 - 1) * 20.0   # sigmoid still gives the same 0/1
                m = mod.calc_metrics(logits_eq, masks)
            else:
                m = mod.calc_metrics(logits, masks)
            for k in keys:
                acc[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def illum_paired(model, loader, device, do_post):
    """CV computed per image across the three illumination levels."""
    model.eval()
    cvs, rb, rd = [], [], []
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            un = torch.clamp(images * 0.5 + 0.5, 0, 1)

            def area(x):
                pr = (torch.sigmoid(model(x)) > 0.5).float()
                if do_post:
                    pr = postprocess(pr)
                return pr.sum(dim=(1, 2, 3)).cpu().numpy()

            a0 = area(images)
            a1 = area((torch.clamp(un * 1.3, 0, 1) - 0.5) / 0.5)
            a2 = area((torch.clamp(un * 0.8, 0, 1) - 0.5) / 0.5)
            trio = np.stack([a0, a1, a2], axis=1)
            cvs.extend((trio.std(axis=1) / (trio.mean(axis=1) + 1e-7)).tolist())
            rb.extend((np.abs(a1 - a0) / (a0 + 1e-7)).tolist())
            rd.extend((np.abs(a2 - a0) / (a0 + 1e-7)).tolist())
    return (float(np.mean(cvs)), float(np.mean(rb)), float(np.mean(rd)))


def run_one(script, run_dir, model_class, do_post, batch_size, out_suffix):
    mod = load_script(script)
    CFG = mod.CONFIG
    device = torch.device(CFG['device'] if torch.cuda.is_available() else 'cpu')
    Cls, cls_name = ((getattr(mod, model_class), model_class) if model_class
                     else find_model_class(mod, script))
    say(f"  model class: {cls_name}   device: {device}")

    # the non-augmented dataset
    ds_eval = mod.ShellDataset(CFG['image_dir'], CFG['mask_dir'],
                               CFG['image_size'], augment=False)
    if CFG.get('subset_size'):
        ds_eval = mod.create_subset(ds_eval, CFG['subset_size'], CFG['seed'])
    total_len = len(ds_eval)
    test_len = int(total_len * CFG.get('test_ratio', 0.15))
    trainval_len = total_len - test_len
    perm = torch.randperm(
        total_len, generator=torch.Generator().manual_seed(CFG['seed'])).tolist()
    tv_idx, te_idx = perm[:trainval_len], perm[trainval_len:]
    say(f"  samples: {total_len} total, {trainval_len} trainval, {test_len} test "
        f"(same seed and split as the training script)")

    test_loader = DataLoader(Subset(ds_eval, te_idx), batch_size=batch_size,
                             shuffle=False, num_workers=0)
    kf = KFold(n_splits=CFG['n_folds'], shuffle=True,
               random_state=CFG['seed'])
    idx = list(range(trainval_len))

    rows = []
    for fold, (_, val_idx) in enumerate(kf.split(idx)):
        ckpt = os.path.join(run_dir, f'fold_{fold}', 'best_iou.pth')
        if not os.path.exists(ckpt):
            say(f"  fold {fold}: {ckpt} not found, skipped")
            continue
        model = Cls().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device),
                              strict=False)
        val_loader = DataLoader(Subset(ds_eval, [tv_idx[i] for i in val_idx]),
                                batch_size=batch_size, shuffle=False,
                                num_workers=0)
        m = eval_loader(mod, model, val_loader, device, do_post)
        cv, rbright, rdark = illum_paired(model, val_loader, device, do_post)
        rec = dict(fold=fold, **{f'val_{k}': v for k, v in m.items()})
        rec.update(best_iou=m['iou'], best_dice=m['dice'],
                   params_M=mod.count_parameters(model),
                   illumination_cv=cv,
                   illum_rel_change_bright=rbright,
                   illum_rel_change_dark=rdark)
        try:
            sp = mod.measure_inference_speed(model, str(device))
            rec.update(fps=sp['fps'], latency_ms=sp['latency_ms'])
        except Exception:
            pass
        rows.append(rec)
        say(f"  fold {fold}: IoU {m['iou']:.4f}  Dice {m['dice']:.4f}  "
            f"HD95 {m['hd95']:.2f}  ASD {m['asd']:.2f}  "
            f"AreaErr {m['area_error_percent']:.2f}%  illumCV {cv:.4f}")
        del model
        torch.cuda.empty_cache()

    if not rows:
        say("  no usable checkpoint; this variant is skipped")
        return None
    R = pd.DataFrame(rows)
    p = os.path.join(run_dir, f'results_summary{out_suffix}.csv')
    R.to_csv(p, index=False)
    say(f"  -> {p}")

    # test set: use the fold with the highest validation IoU
    best = int(R.loc[R.best_iou.idxmax(), 'fold'])
    ckpt = os.path.join(run_dir, f'fold_{best}', 'best_iou.pth')
    model = Cls().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    tm = eval_loader(mod, model, test_loader, device, do_post)
    pt = os.path.join(run_dir, f'test_results{out_suffix}.csv')
    pd.DataFrame([tm]).to_csv(pt, index=False)
    say(f"  test set (fold {best}): IoU {tm['iou']:.4f}  Dice {tm['dice']:.4f}"
        f"  HD95 {tm['hd95']:.2f}")
    say(f"  -> {pt}")
    del model
    torch.cuda.empty_cache()
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script")
    ap.add_argument("--run_dir")
    ap.add_argument("--all", nargs="+",
                    help='one or more entries of the form "name=script=run_dir"')
    ap.add_argument("--model_class", default=None)
    ap.add_argument("--no_postprocess", action="store_true",
                    help="skip post-processing; apply only the augmentation and illumination fixes")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--suffix", default="_reeval")
    a = ap.parse_args()
    do_post = not a.no_postprocess

    say(f"post-processing: "
        f"{'on (5x5 closing + largest component)' if do_post else 'off'}")
    say("evaluation-time augmentation: off\n")

    jobs = []
    if a.all:
        for spec in a.all:
            parts = spec.split("=")
            if len(parts) != 3:
                raise SystemExit(f'expected "name=script=run_dir", got {spec!r}')
            jobs.append(tuple(parts))
    elif a.script and a.run_dir:
        jobs.append((os.path.basename(a.script), a.script, a.run_dir))
    else:
        raise SystemExit("use --script with --run_dir, or --all")

    for name, script, run_dir in jobs:
        say("=" * 70)
        say(f"{name}   ({script})")
        say("=" * 70)
        if not os.path.exists(script):
            say(f"  script not found, skipped")
            continue
        if not os.path.isdir(run_dir):
            say(f"  run directory not found, skipped: {run_dir}")
            continue
        try:
            run_one(script, run_dir, a.model_class, do_post, a.batch_size,
                    a.suffix)
        except Exception as e:
            say(f"  failed: {type(e).__name__}: {e}")
        say("")

    say("Done. When producing tables and figures, point seg_make_tables_figures.py")
    say(f"at results_summary{a.suffix}.csv, or pass --suffix.")


if __name__ == "__main__":
    main()
