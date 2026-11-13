#!/usr/bin/env bash
# Push the working tree (minus build outputs) to ~/aq/src on the Spark.
set -euo pipefail
cd "$(dirname "$0")/.."
COPYFILE_DISABLE=1 tar --no-xattrs -czf - --exclude=./target --exclude=./.git --exclude=./.claude --exclude='*.safetensors' . \
  | ssh -o BatchMode=yes "$AQ_REMOTE" 'mkdir -p ~/aq/src && tar -xzf - -C ~/aq/src 2>/dev/null && echo synced to spark'
