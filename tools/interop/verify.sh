#!/usr/bin/env bash
# Round-trip interop check between ri and pi.
#
#   tools/interop/verify.sh [pi-repo]
#
# 1. ri reads pi-written fixtures (cargo test, runs in CI too).
# 2. ri writes a session; pi's own storage code reads it back and must agree
#    with what ri reports. This half needs a pi checkout with `npm ci` done
#    and `packages/ai` built, so it lives here rather than in cargo test.
set -euo pipefail

PI_REPO="${1:-${PI_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)/pi}}"
RI_REPO="$(cd "$(dirname "$0")/../.." && pwd)"

if [ ! -d "$PI_REPO/packages/agent/src" ]; then
  echo "pi checkout not found at $PI_REPO (pass the path as \$1)" >&2
  exit 1
fi

echo "==> ri reads pi-written fixtures"
cargo test --manifest-path "$RI_REPO/Cargo.toml" -p ri-agent-core \
  --test interop_pi_fixtures --test interop_write_for_pi

echo "==> pi reads the ri-written session"
DUMP="$(cd "$RI_REPO/tools/interop" && bun dump-pi-session.ts "$PI_REPO" \
  "$RI_REPO/target/interop/ri-session-v3.jsonl")"
echo "$DUMP" | python3 -c '
import json, sys
dump = json.load(sys.stdin)
expected_types = [
    "message", "model_change", "thinking_level_change", "active_tools_change",
    "message", "message", "message", "custom_message", "compaction", "label",
    "session_info",
]
assert dump["entryTypes"] == expected_types, dump["entryTypes"]
# pi stops the branch at the compaction boundary and keeps firstKeptEntryId.
assert len(dump["branchIds"]) == 5, dump["branchIds"]
context = dump["context"]
assert context["messageRoles"] == ["compactionSummary", "user", "custom"], context
assert context["messageTexts"][1] == "Keep this one", context
assert context["messageTexts"][2] == "side note", context
print("pi agrees with the ri-written session")
'

echo "==> regenerating fixtures is a separate, deliberate step:"
echo "    bun tools/interop/generate-pi-fixtures.ts $PI_REPO crates/ri-agent-core/tests/fixtures"
