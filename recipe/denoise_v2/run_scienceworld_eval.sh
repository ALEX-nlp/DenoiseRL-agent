#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/../.."
exec python3 -m recipe.denoise_v2.task_suite.launch --benchmark scienceworld --method denoise --mode eval "$@"
