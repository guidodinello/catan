#!/usr/bin/env bash
# Experiment 014 Stage 1 orchestrator. Resumable: a job whose DONE marker exists is
# skipped, so after a crash or power loss only the in-flight job is redone (BC saves
# its checkpoint only at the end, so a half-run leaves nothing to resume from).
# One tmux session per job (catan014-bc-<name>), sequential (one GPU), guards before
# each launch, timeout 8h, no pkill. Usage: stage1.sh sweep | final
set -u
R=/home/guido/projects/catan/rl_runs/exp014
W=/home/guido/projects/catan-014-run
V=/media/guido/0DF7128F0DF7128F/catan-014-venv
PY=$V/bin/python
TOOLS=$R/helpers/stage1_tools.py
mkdir -p $R/bc $R/logs $R/done

guards() {  # refuse to launch on top of anything
  if pgrep -af '[r]l\.train|[r]l\.bc|[e]xperiments\.(benchmark|search_eval|exp014)' | grep -v pgrep | grep -q .; then
    echo "GUARD: a training/benchmark process is running; refusing" >&2; return 1; fi
  if tmux ls 2>/dev/null | grep -q '^catan014-bc-'; then
    echo "GUARD: a catan014-bc session exists; refusing" >&2; return 1; fi
  free -g | sed -n 2p
}

run_job() {  # run_job <name> <args to rl.bc train ...>
  local name=$1; shift
  if [ -e $R/done/$name ]; then echo "skip $name (done)"; return 0; fi
  guards || exit 1
  local log=$R/logs/$name.log
  echo "launch $name $(date +%T)"
  tmux new -d -s catan014-bc-$name "bash -c 'cd $W && PYTHONPATH=$W OMP_NUM_THREADS=1 timeout 8h $PY -m rl.bc train $* --device cuda --sanity-episodes 0 > >(tee $log) 2>&1; rc=\$?; echo EXIT=\$rc >> $log; [ \$rc = 0 ] && touch $R/done/$name'"
  sleep 5
  while tmux has-session -t catan014-bc-$name 2>/dev/null; do sleep 15; done
  if [ ! -e $R/done/$name ]; then echo "FAILED $name (see $log)" >&2; exit 1; fi
  echo "done $name $(date +%T)"
}

common="--epochs 30 --patience 3 --batch-size 1024 --select-split tune"
case ${1:-} in
  sweep)
    if [ ! -e $R/done/mksweep ]; then
      guards || exit 1
      $PY $TOOLS make-sweep-ds --src /home/guido/projects/catan/rl_runs/bc/v1 --dst $R/bc_sweep_ds \
        > $R/logs/make_sweep_ds.log 2>&1 && touch $R/done/mksweep || { echo FAILED mksweep >&2; exit 1; }
    fi
    for lr in 1e-3 3e-4; do
      for arm in mlp graph; do
        run_job sweep_${arm}_lr$lr "--dataset $R/bc_sweep_ds --save $R/bc/sweep_${arm}_lr$lr.zip $common --lr $lr --seed 0 --policy $arm"
      done
    done
    $PY $TOOLS choose-lr --logdir $R/logs --out $R/lr_choice.json || exit 1
    ;;
  final)
    [ -e $R/lr_choice.json ] || { echo "no lr_choice.json; run sweep first" >&2; exit 1; }
    for seed in 0 1 2; do
      for arm in mlp graph; do
        lr=$($PY -c "import json;print(json.load(open('$R/lr_choice.json'))['$arm']['lr'])")
        run_job final_${arm}_s$seed "--dataset /home/guido/projects/catan/rl_runs/bc/v1 --save $R/bc/${arm}_s$seed.zip $common --lr $lr --seed $seed --policy $arm"
      done
    done
    ;;
  *) echo "usage: $0 sweep|final" >&2; exit 2 ;;
esac
echo "STAGE1 PHASE ${1} COMPLETE $(date +%T)"
