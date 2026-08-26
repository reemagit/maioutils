#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Publish local maioutils changes to GitHub.

Usage:
  ./scripts/publish.sh [--yes] [--dry-run] ["commit message"]

Options:
  -y, --yes      Skip the final confirmation prompt.
  -n, --dry-run  Run checks and show changes without committing or pushing.
  -h, --help     Show this help message.
EOF
}

confirm=false
dry_run=false
commit_message=""

while (($#)); do
    case "$1" in
        -y|--yes)
            confirm=true
            ;;
        -n|--dry-run)
            dry_run=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -n "$commit_message" ]]; then
                echo "Error: provide the commit message as one quoted argument." >&2
                usage >&2
                exit 2
            fi
            commit_message="$1"
            ;;
    esac
    shift
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/.." rev-parse --show-toplevel)"
cd "$repo_root"

branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
    echo "Error: cannot publish while Git is in detached-HEAD state." >&2
    exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
    echo "Error: this repository has no Git remote named 'origin'." >&2
    exit 1
fi

python="$repo_root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Error: development environment not found at $repo_root/.venv" >&2
    echo "Create it with: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
    exit 1
fi

echo "Running tests before publishing..."
# This small shim avoids a known native-readline crash in the local Miniconda
# interpreter while leaving the package tests themselves unchanged.
"$python" -c 'import sys, types; sys.modules["readline"] = types.ModuleType("readline"); import pytest; raise SystemExit(pytest.main(["-q"]))'

if [[ -z "$(git status --porcelain)" ]]; then
    echo "Nothing to publish: the working tree is clean."
    exit 0
fi

echo
echo "Changes that will be published:"
git status --short

if [[ "$dry_run" == true ]]; then
    echo
    echo "Dry run complete; nothing was committed or pushed."
    exit 0
fi

if [[ -z "$commit_message" ]]; then
    read -r -p "Commit message: " commit_message
fi

if [[ -z "$commit_message" ]]; then
    echo "Error: commit message cannot be empty." >&2
    exit 1
fi

if [[ "$confirm" != true ]]; then
    read -r -p "Commit and push these changes to origin/$branch? [y/N] " answer
    case "$answer" in
        y|Y|yes|YES)
            ;;
        *)
            echo "Publish cancelled; no Git changes were made."
            exit 0
            ;;
    esac
fi

git add --all
git commit -m "$commit_message"
git pull --rebase origin "$branch"
git push --set-upstream origin "$branch"

echo "Published successfully to origin/$branch."
