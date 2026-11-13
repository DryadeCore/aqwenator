#!/usr/bin/env bash
# Run a command on the DGX Spark. Usage: scripts/spark.sh 'cmd'
exec ssh -o BatchMode=yes "$AQ_REMOTE" "$*"
