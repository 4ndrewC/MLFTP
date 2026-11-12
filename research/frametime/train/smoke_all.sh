#!/usr/bin/env bash
# End-to-end check of the phase-2 code WITHOUT training anything:
#   unit tests (synthetic arrays) + every entry point with --dry-run (1 session per split, <= 2000 frames,
#   random/untrained weights) writing into a scratch folder, checkpoint validation, the engine loader
#   (build/spkm_eval check) on every exported .spkm, and aggregate.py over the scratch results.
#
#   research/frametime/train/smoke_all.sh [SCRATCH_DIR]      (default: $TMPDIR/spark-ft-dryrun)
#
# Runs on the laptop in ~1 minute (torch-based parts skip if torch is missing; REQUIRE_TORCH=1 makes a
# missing torch a failure, which the cluster gate uses) and on the cluster (cluster/jobs/train_dryrun.slurm).
# Never writes to research/frametime/checkpoints or results. The no-fit guard is ON for the whole script
# (FRAMETIME_NO_FIT=1): every fitting call site refuses, so dry runs export untrained weights only.
set -euo pipefail
export FRAMETIME_NO_FIT=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FT="$(dirname "$HERE")"
ROOT="$(cd "$FT/../.." && pwd)"
OUT="${1:-${TMPDIR:-/tmp}/spark-ft-dryrun}"
PY="${PYTHON:-python3}"
case "$(cd "$(dirname "$OUT")" 2>/dev/null && pwd)/$(basename "$OUT")" in
  "$FT"/checkpoints*|"$FT"/results*|"$FT") echo "refusing to write dry-run output into $OUT" >&2; exit 1 ;;
esac
rm -rf "$OUT"; mkdir -p "$OUT"
COMMON=(--dry-run --out-root "$OUT" --cache-dir "$OUT/cache" --workers 1)
HAVE_TORCH=$($PY -c 'import torch' 2>/dev/null && echo 1 || echo 0)
HAVE_LGB=$($PY -c 'import lightgbm' 2>/dev/null && echo 1 || echo 0)
echo "== scratch: $OUT  (torch=$HAVE_TORCH lightgbm=$HAVE_LGB no-fit guard=$FRAMETIME_NO_FIT)"
if [[ ${REQUIRE_TORCH:-0} == 1 && $HAVE_TORCH == 0 ]]; then
  echo "REQUIRE_TORCH=1 but torch is not importable: the torch models would not be checked" >&2; exit 1
fi

echo "== unit tests"
$PY "$HERE/tests/test_train_pipeline.py" 2>&1 | tail -5

echo "== data prep (cache + counterfactual rows)"
$PY "$HERE/data.py" "${COMMON[@]}" --cf-ladder --cf-levers

echo "== baselines"
$PY "$HERE/baselines.py" "${COMMON[@]}"

if [[ $HAVE_TORCH == 1 ]]; then
  echo "== train.py (every experiment, dry run: untrained weights)"
  $PY "$HERE/train.py" "${COMMON[@]}" --device cpu $($PY -c "
import sys; sys.path.insert(0, '$HERE'); import experiments as E; print(' '.join(E.EXPERIMENTS))")
else
  echo "== train.py (torch missing: GBT and RLS experiments only)"
  $PY "$HERE/train.py" "${COMMON[@]}" gbt_q f_gbt_hist rls_hybrid
fi

echo "== importance"
$PY "$HERE/importance.py" "${COMMON[@]}" --perm-rows 2000

echo "== controller simulation"
$PY "$HERE/evaluate.py" controller "${COMMON[@]}"

echo "== validator + engine loader on every exported checkpoint"
n=0; bad=0
for d in "$OUT"/checkpoints/*/; do
  n=$((n + 1))
  if ! $PY "$FT/tools/validate_checkpoint.py" "$d" >/dev/null; then
    echo "FAIL validate $d"; $PY "$FT/tools/validate_checkpoint.py" "$d" || true; bad=$((bad + 1))
  fi
  if [[ -x "$ROOT/build/spkm_eval" ]]; then
    "$ROOT/build/spkm_eval" check "$d/model.spkm" | sed "s|^|  $(basename "$d"): |" || bad=$((bad + 1))
  fi
done
[[ -x "$ROOT/build/spkm_eval" ]] || echo "  (build/spkm_eval not built: run 'make parity' to also check the engine loader)"
echo "  $n checkpoints, $bad failures"

echo "== aggregate"
$PY "$HERE/aggregate.py" --out-root "$OUT" --include-dry
[[ $bad == 0 ]] && echo "SMOKE OK ($OUT)" || { echo "SMOKE FAILED"; exit 1; }
