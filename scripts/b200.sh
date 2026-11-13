#!/usr/bin/env bash
# Run a command on the B200 box. Usage: scripts/b200.sh 'cmd'
exec ssh "$AQ_REMOTE" "$*"
