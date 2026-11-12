#!/usr/bin/env bash
# Full teacher data pipeline. Every step is resumable and cached; spend is capped by --budget-usd.
set -euo pipefail
cd "$(dirname "$0")/.."
BUDGET="${BUDGET:-18}"
CALLS="${CALLS:-1100}"
THINK="${THINK:-5000}"
run() { echo "=== $(date '+%F %T') $*"; uv run --no-sync python -m "$@"; }
[[ -s artifacts/data/public.jsonl ]] || run data.convert_public
run data.gen_scenarios --calls "$CALLS" --workers 32 --budget-usd "$BUDGET" --offpeak-only
run data.label_fast --orders 3 --workers 48 --budget-usd "$BUDGET" --offpeak-only
run data.label_think --n "$THINK" --effort high --workers 48 --budget-usd "$BUDGET" --offpeak-only
run data.build_splits
echo "=== $(date '+%F %T') done"
sqlite3 artifacts/cache/teacher.sqlite "select step, count(*), round(sum(cost),4) from ledger group by step"
