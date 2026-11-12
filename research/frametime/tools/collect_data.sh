#!/usr/bin/env bash
# Collect frame-time data into research/frametime/data-v2/ (sessions/*.sfl, fingerprints/, MANIFEST.json).
#
#   tools/collect_data.sh [--plan bulk,solo,matched,levers,vhw] [--max-minutes 30] [--max-mb 300]
#                         [--cpus 0,2,4,6,8,12 | auto:N] [--solo-cpu 2] [--ecore-cpu 10] [--paced-frac 0.6]
#                         [--out DIR] [--dataset NAME] [--phase A] [--plan-seed N]
#
# Plans: bulk (one pinned worker per CPU, random sessions), solo (one game at a time), matched
# (same sessions solo, then all at once), levers (paired single-lever sweeps), vhw (co-runner
# virtual hardware, P vs E core). Details: python3 tools/collect_data.py --help.
#
# A hardware fingerprint is measured once per (CPU, co-runner) before its first session and
# embedded in every session header. Refuses to run on battery (AC or no battery = OK); waits for
# other heavy processes first; writes research/frametime/data/.collecting while running.
# Caps: --max-minutes is wall time for this invocation, --max-mb the bytes it may write. Run it in
# the background; don't compile at the same time. Summarise with tools/summarize_data.py DIR.
# Collection runs the game; it never trains anything. Phase-1 v1 data stays in data/ untouched.
#
# Optional real windowed play (opens a window; run it yourself, the agent never does):
#   build/spark_warden --mute --log my_play.sfl --stats      (play; F3 overlay)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
[[ "${1:-}" == -h || "${1:-}" == --help ]] && { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
[[ -x "$ROOT/build/spark_warden" ]] || { echo "error: build the game first (make)" >&2; exit 1; }
command -v taskset >/dev/null || { echo "error: taskset not found (util-linux)" >&2; exit 1; }

echo "== power: $(cat /sys/class/power_supply/ADP1/online 2>/dev/null | sed 's/1/AC/;s/0/battery/' || echo unknown)"
echo "== disk before: $(df -h "$ROOT" | awk 'NR==2 {print $4 " free of " $2}')"
python3 "$HERE/collect_data.py" "$@"
echo "== disk after:  $(df -h "$ROOT" | awk 'NR==2 {print $4 " free of " $2}')"
du -sh "$ROOT"/research/frametime/data* 2>/dev/null || true
