#!/usr/bin/env bash
# End-to-end: build the panel from the raw files, run every regime at one gamma, sweep gamma for R2.
set -euo pipefail
RAW=${RAW:-data/raw}
OUT=${OUT:-results}
GAMMA=${GAMMA:-0.3}
python -m eq_model check-load --load $RAW/load
python -m eq_model build-panel --load $RAW/load --gen $RAW/gen_by_fuel \
    --capacity $RAW/capacity/vre_capacity.csv \
    --eia-active $RAW/capacity/generators_active_07_2026.csv \
    --eia-retired $RAW/capacity/generators_retired_07_2026.csv \
    --save-capacity data/vre_capacity_with_coverage.csv --out data/panel.npz
python -m eq_model run   --panel data/panel.npz --regimes P1,P2,R2,R3,R4,R5,R6 --gamma $GAMMA --out $OUT/gamma$GAMMA
python -m eq_model sweep --panel data/panel.npz --regime R2 --gammas 0,0.1,0.25,0.5,1.0 --out $OUT/sweep_R2
