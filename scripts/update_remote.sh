#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Update and reinstall maioutils on a remote machine.

Usage:
  ./scripts/update_remote.sh [VENV_OR_PYTHON]

VENV_OR_PYTHON may be either a virtual-environment directory or the full path
to its Python executable. If omitted, the script uses the active virtualenv or
Conda environment, then falls back to python or python3 from the current PATH.

Optional environment variables:
  MAIOUTILS_REMOTE  Git remote to pull from (default: origin)
  MAIOUTILS_BRANCH  Branch to update (default: current branch)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if (($# > 1)); then
    usage >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/.." rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: the remote checkout contains local changes:" >&2
    git status --short >&2
    echo "Resolve or remove them before updating." >&2
    exit 1
fi

remote="${MAIOUTILS_REMOTE:-origin}"
branch="${MAIOUTILS_BRANCH:-$(git branch --show-current)}"

if [[ -z "$branch" ]]; then
    echo "Error: set MAIOUTILS_BRANCH because Git is in detached-HEAD state." >&2
    exit 1
fi

if (($# == 1)); then
    if [[ -d "$1" ]]; then
        python="$1/bin/python"
    else
        python="$1"
    fi
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    python="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    python="$CONDA_PREFIX/bin/python"
elif command -v python >/dev/null 2>&1; then
    python="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    python="$(command -v python3)"
else
    echo "Error: no Python executable was found in the current environment." >&2
    echo "Activate Conda/a virtualenv or pass its path explicitly." >&2
    exit 1
fi

if [[ ! -x "$python" ]]; then
    echo "Error: Python executable not found at $python" >&2
    echo "Activate the target virtual environment or pass its path explicitly." >&2
    exit 1
fi

echo "Updating $repo_root from $remote/$branch..."
git pull --ff-only "$remote" "$branch"

echo "Installing the updated package with $python..."
"$python" -m pip install --upgrade "$repo_root"

"$python" -c 'from importlib.metadata import version; from maioutils import make_synthetic_dataframe; print("maioutils {} is ready".format(version("maioutils")))'

echo "Update complete. Restart any running Python process or Jupyter kernel that uses maioutils."
