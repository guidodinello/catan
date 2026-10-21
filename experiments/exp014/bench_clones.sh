#!/usr/bin/env bash
# Exp 014 secondary (d): seed-0 clone of each arm, n=4000 on fresh_014, rl_vs_heuristic.
set -u
R=/home/guido/projects/catan/rl_runs/exp014; W=/home/guido/projects/catan-014-run
PY=/media/guido/0DF7128F0DF7128F/catan-014-venv/bin/python
WORKERS=${1:?workers}
cd $W && export PYTHONPATH=$W OMP_NUM_THREADS=1
for arm in mlp graph; do
  echo "start $arm $(date +%T)"
  timeout 2h $PY -m experiments.benchmark --mode rl_vs_heuristic --games 4000 --players 4 \
    --engine-seed-base 21000001 --driver-seed-base 21000001 --workers $WORKERS \
    --checkpoint $R/bc/${arm}_s0.zip || { echo "FAILED $arm"; exit 1; }
  mv $W/experiments/results/benchmark_rl_vs_heuristic_p4_${arm}_s0.json $R/bench/ || exit 1
  echo "done $arm $(date +%T)"
done
echo BENCH COMPLETE $(date +%T)
