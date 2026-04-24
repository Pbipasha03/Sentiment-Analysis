#!/bin/bash

# Backward-compatible entry point. The real launcher lives in START_SIMPLE.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/START_SIMPLE.sh" "$@"
