#!/usr/bin/env bash
# Exp 014 Stage 2 primary-metric benchmarks: each fixed <label>_3000000.zip once, n=4000 on
# fresh_014 (engine/driver seeds 21000001..), rl_vs_heuristic, CPU only. Same command as
# bench_clones.sh. Skip-if-done per checkpoint (result JSON in bench/). Sequential.
# Narrower guard than the registered one: a live rl.train (graph s4) is expected and left alone.
# Stop: touch $R/bench/STOP (checked between checkpoints) or SIGTERM the session's python.
# Usage: bench_stage2.sh <workers> <mem_budget_GB> <arm>:<seed> ...   e.g. 2 1.8 mlp:3 mlp:4 graph:3
set -u
R=/home/guido/projects/catan/rl_runs/exp014
W=/home/guido/projects/catan-014-run
PY=/media/guido/0DF7128F0DF7128F/catan-014-venv/bin/python
WORKERS=${1:?workers}; BUDGET=${2:?mem budget GB}; shift 2
SMOKE=${SMOKE:-}   # smoke mode: SMOKE="<engine_seed_base> <games> <checkpoint>" (never fresh_014)
[ -n "$SMOKE" ] || [ $# -gt 0 ] || { echo "usage: $0 <workers> <mem_budget_GB> <arm>:<seed> ..." >&2; exit 2; }
mkdir -p $R/bench $R/done $R/logs

guards() {
  if pgrep -af '[e]xperiments\.benchmark' | grep -v pgrep | grep -q .; then
    echo "GUARD: a benchmark process is already running; refusing" >&2; return 1; fi
  local me; me=$(tmux display-message -p '#S' 2>/dev/null || true)
  if tmux ls -F '#S' 2>/dev/null | grep '^catan014-bench-' | grep -qv "^${me}$"; then
    echo "GUARD: another catan014-bench session exists; refusing" >&2; return 1; fi
  return 0
}

bench() {  # bench <seed_base> <games> <checkpoint> -> result JSON path in the worktree
  local sb=$1 games=$2 ckpt=$3
  ( cd $W && export PYTHONPATH=$W OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
    exec systemd-run --user --scope --quiet -p MemoryMax=${BUDGET}G -p MemorySwapMax=0 \
      choom -n 1000 -- timeout 2h $PY -m experiments.benchmark --mode rl_vs_heuristic \
      --games $games --players 4 --engine-seed-base $sb --driver-seed-base $sb \
      --workers $WORKERS --checkpoint $ckpt )
}

if [ -n "$SMOKE" ]; then
  set -- $SMOKE
  guards || exit 1
  echo "smoke start $(date +%T) seed_base=$1 games=$2 workers=$WORKERS"
  bench "$1" "$2" "$3" || { echo "SMOKE FAILED" >&2; exit 1; }
  echo "smoke done $(date +%T)"; exit 0
fi

for spec in "$@"; do
  arm=${spec%%:*}; seed=${spec##*:}
  label=catan014_ft_${arm}_s$seed
  ckpt=$R/selfplay/$label/${label}_3000000.zip
  out=$R/bench/benchmark_rl_vs_heuristic_p4_${label}_3000000.json
  [ -e $R/bench/STOP ] && { echo "STOP file present; halting"; exit 0; }
  if [ -e "$out" ]; then echo "skip $label (done)"; continue; fi
  [ -e $R/done/ft_${arm}_s$seed ] && [ -e "$ckpt" ] || { echo "FAILED $label: no done marker/checkpoint" >&2; exit 1; }
  guards || exit 1
  avail=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  awk -v a="$avail" -v b="$BUDGET" 'BEGIN{exit !(a >= b + 2.0)}' || { echo "GUARD: ${avail} GB available < budget ${BUDGET} + 2 GB headroom" >&2; exit 1; }
  echo "start $label $(date +%T) workers=$WORKERS avail=${avail}GB"
  bench 21000001 4000 "$ckpt" || { echo "FAILED $label" >&2; exit 1; }
  mv $W/experiments/results/benchmark_rl_vs_heuristic_p4_${label}_3000000.json "$out" || exit 1
  touch $R/done/bench_${arm}_s$seed
  echo "done $label $(date +%T)"
done
echo "BENCH BATCH COMPLETE $(date +%T)"
