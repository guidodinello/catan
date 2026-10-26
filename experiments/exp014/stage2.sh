#!/usr/bin/env bash
# Experiment 014 Stage 2 orchestrator: 3M-step PPO fine-tunes from the seed-0 BC clones,
# exactly as log 014 Design > Stage 2. Sequential (one GPU), one tmux session per job
# (catan014-ft-<arm>-s<seed>), guards before each launch, timeout 8h, no pkill.
# A job whose done marker exists is skipped; a failed job stops the batch.
# Usage: stage2.sh mlp:3 graph:3 mlp:4 ...
set -u
R=/home/guido/projects/catan/rl_runs/exp014
W=/home/guido/projects/catan-014-run
V=/media/guido/0DF7128F0DF7128F/catan-014-venv
PY=$V/bin/python
mkdir -p $R/logs $R/done

rate_for() {  # fresh_014 n=4000 win rate of the seed-0 clone (log 014 Result)
  case $1 in mlp) echo 0.107 ;; graph) echo 0.25025 ;; *) return 1 ;; esac
}

guards() {  # hard refusals, then a RAM wait (5 GB available, up to 30 min)
  if pgrep -af '[r]l\.train|[r]l\.bc|[e]xperiments\.' | grep -v pgrep | grep -q .; then
    echo "GUARD: a training/benchmark process is running; refusing" >&2; return 1; fi
  if tmux ls 2>/dev/null | grep -q '^catan014-ft-'; then
    echo "GUARD: a catan014-ft session exists; refusing" >&2; return 1; fi
  local i avail
  for i in $(seq 30); do
    avail=$(free -g | awk 'NR==2{print $7}')
    [ "$avail" -ge 5 ] && { free -g | sed -n 2p; return 0; }
    echo "GUARD: only ${avail} GB available, waiting ($i/30)" >&2; sleep 60
  done
  echo "GUARD: RAM stayed below 5 GB for 30 min; refusing" >&2; return 1
}

run_job() {  # run_job <arm> <seed>
  local arm=$1 seed=$2 name=ft_$1_s$2 label=catan014_ft_$1_s$2
  if [ -e $R/done/$name ]; then echo "skip $name (done)"; return 0; fi
  guards || exit 1
  local rate; rate=$(rate_for $arm) || exit 1
  local log=$R/logs/$name.log
  echo "launch $name $(date +%T)"
  tmux new -d -s catan014-ft-$arm-s$seed "bash -c 'cd $W && PYTHONPATH=$W OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 timeout 8h $PY -m rl.train --bc-init $R/bc/${arm}_s0.zip --bc-init-rate $rate --baseline-mix 0.5 --ent-coef 0.01 --learning-rate 1e-4 --envs 8 --n-steps 512 --device cuda --seed $seed --eval-opponents heuristic --eval-every 250000 --eval-episodes 200 --regression-margin 0.10 --regression-patience 1000 --steps 3000000 --selfplay-dir $R/selfplay --run-dir $R/$label --label $label > >(tee $log) 2>&1; rc=\$?; echo EXIT=\$rc >> $log; [ \$rc = 0 ] && touch $R/done/$name'"
  sleep 5
  while tmux has-session -t catan014-ft-$arm-s$seed 2>/dev/null; do sleep 30; done
  if [ ! -e $R/done/$name ]; then echo "FAILED $name (see $log)" >&2; exit 1; fi
  echo "done $name $(date +%T)"
}

[ $# -gt 0 ] || { echo "usage: $0 <arm>:<seed> ..." >&2; exit 2; }
for spec in "$@"; do run_job "${spec%%:*}" "${spec##*:}"; done
echo "STAGE2 BATCH COMPLETE $(date +%T)"
