#!/usr/bin/env bash
# End-to-end smoke test on synthetic data. No study data is required.
#
# Runs both material types so that capability detection can be seen doing its
# job: the outbred panel keeps every module, the inbred panel has its dominance
# module blocked before anything is fitted.
set -euo pipefail
cd "$(dirname "$0")"
ADEPT=../adept

echo "== 1. synthetic data =="
python make_synthetic.py --out demo_out --mode outbred --n 300 --m 1200
python make_synthetic.py --out demo_in  --mode inbred  --n 300 --m 1200

echo
echo "== 2. capability detection =="
python $ADEPT/capability_audit.py --bfile demo_out --out cap_outbred
python $ADEPT/capability_audit.py --bfile demo_in  --out cap_inbred

echo
echo "== 3. one trait, end to end =="
python $ADEPT/trait_epistasis_to_allocation.py \
    --bfile demo_out --trait demo_out \
    --top_k 5 --folds 3 --jobs 2 --threads_per_job 2 \
    --n_candidates 40 --n_crosses 15 \
    --cache_dir kc_out --out pipe_demo_out

python $ADEPT/trait_epistasis_to_allocation.py \
    --bfile demo_in --trait demo_in \
    --top_k 5 --folds 3 --jobs 2 --threads_per_job 2 \
    --n_candidates 40 --n_crosses 15 \
    --cache_dir kc_in --out pipe_demo_in

echo
echo "== 4. collect and plot =="
python $ADEPT/collect_allocation_results.py --species pig \
    --alloc_glob "pipe_demo_*" --pairs_glob "pipe_demo_*" --out tables
python $ADEPT/make_allocation_figures.py --tables tables --out figures

echo
echo "Expected: cap_inbred/capability_profile.json reports dominance blocked,"
echo "and pipe_demo_in has no ADD+DOM row in its strategy summary."
