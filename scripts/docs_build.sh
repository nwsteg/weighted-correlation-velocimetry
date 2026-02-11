#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "serve" ]]; then
  mkdocs serve
else
  mkdocs build --strict
fi
