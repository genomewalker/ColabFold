#!/usr/bin/env bash
# Apply all alphafold model patches to the alphafold-colabfold site-packages.
# Run after installing the colabfold conda/pip environment.
#
# Usage:
#   bash patches/apply_alphafold_patches.sh [PYTHON_EXECUTABLE]
#
# PYTHON_EXECUTABLE defaults to 'python3'. Pass the full path to the env's
# Python if the env is not active, e.g.:
#   bash patches/apply_alphafold_patches.sh /opt/colabfold-env/bin/python3

set -euo pipefail

PYTHON="${1:-python3}"

AF_SITE=$("$PYTHON" -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('alphafold')
print(pathlib.Path(spec.origin).parent)
")

if [ -z "$AF_SITE" ]; then
    echo "ERROR: alphafold package not found in $PYTHON environment."
    exit 1
fi

echo "alphafold model dir: $AF_SITE/model"
echo "alphafold-colabfold version: $("$PYTHON" -c "import importlib.metadata; print(importlib.metadata.version('alphafold-colabfold'))")"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/alphafold"

apply() {
    local patchfile="$SCRIPT_DIR/$1"
    local target="$AF_SITE/model/${1%.patch}"
    echo "Patching $target ..."
    if patch --dry-run -p1 -N "$target" < "$patchfile" >/dev/null 2>&1; then
        patch -p1 -N "$target" < "$patchfile"
        echo "  OK"
    else
        # Check if already applied
        if patch --dry-run -p1 -R "$target" < "$patchfile" >/dev/null 2>&1; then
            echo "  Already applied — skipping"
        else
            echo "  ERROR: patch failed — check for conflicts"
            exit 1
        fi
    fi
}

apply model.py.patch
apply modules.py.patch
apply config.py.patch

echo ""
echo "All patches applied successfully."
