#!/bin/bash
# Verify the codebase is black-formatted without modifying it.
# Exits non-zero (and prints a diff) if any file needs reformatting.
set -e

cd "$(dirname "$0")/.."

echo "Checking formatting with black..."
uv run black --check --diff backend main.py
