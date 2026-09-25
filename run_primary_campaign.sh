#!/usr/bin/env bash
set -euo pipefail
for seed in 0 1 2 3 4; do
  python experiments_campaign_spectral.py \
    --campaign --seed "$seed" --outdir "results/seed_$seed" \
    --dim 20 --forcing 8 --dt 0.01 --data-steps 20000 --burnin 2000 \
    --hidden 128 --epochs 20 --batch-size 512 --lr 1e-3 \
    --horizons "1,2,5,10,20,40,80,160" \
    --n-test-states 100 --n-pgd-states 20 \
    --perturb-horizon 20 --perturb-epsilon 1e-2 \
    --pgd-steps 100 --restarts 5 \
    --eps-min -6 --eps-max -1 --eps-count 10 --random-directions 20
done
