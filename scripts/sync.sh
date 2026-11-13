#!/usr/bin/env bash
# Push the working tree (minus build outputs) to the B200.
# Remote dir defaults to ~/aq/src; set AQ_REMOTE_DIR (relative to ~) for a separate checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=${AQ_REMOTE_DIR:-aq/src}
COPYFILE_DISABLE=1 tar --no-xattrs -czf - --exclude=./target --exclude=./.git --exclude='*.safetensors' --exclude=./bench-results . \
  | ssh "$AQ_REMOTE" "mkdir -p ~/$REMOTE && tar -xzf - -C ~/$REMOTE 2>/dev/null && echo synced to ~/$REMOTE"
