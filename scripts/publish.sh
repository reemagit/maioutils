#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Publish local maioutils changes to GitHub.

Usage:
  ./scripts/publish.sh [--yes] [--dry-run] [--bump LEVEL|--no-bump] ["commit message"]

Options:
  -y, --yes      Skip the final confirmation prompt.
  -n, --dry-run  Run checks and show changes without committing or pushing.
  -b, --bump     Increment the version: patch, minor, or major.
      --no-bump  Keep the current version without prompting.
  -h, --help     Show this help message.
EOF
}

confirm=false
dry_run=false
bump_level=""
bump_choice_set=false
commit_message=""

while (($#)); do
    case "$1" in
        -y|--yes)
            confirm=true
            ;;
        -n|--dry-run)
            dry_run=true
            ;;
        -b|--bump)
            if (($# < 2)); then
                echo "Error: --bump requires patch, minor, or major." >&2
                exit 2
            fi
            bump_level="$2"
            bump_choice_set=true
            shift
            ;;
        --bump=*)
            bump_level="${1#*=}"
            bump_choice_set=true
            ;;
        --no-bump)
            bump_level=""
            bump_choice_set=true
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

case "$bump_level" in
    ""|patch|minor|major)
        ;;
    *)
        echo "Error: version bump must be patch, minor, or major." >&2
        exit 2
        ;;
esac

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

if [[ "$bump_choice_set" != true ]]; then
    if [[ ! -t 0 ]]; then
        echo "Error: version selection requires an interactive terminal." >&2
        echo "Use --bump patch|minor|major or --no-bump." >&2
        exit 2
    fi

    echo "Version bump:"
    echo "  0) none (default)"
    echo "  1) patch"
    echo "  2) minor"
    echo "  3) major"
    read -r -p "Select [0]: " bump_choice

    case "${bump_choice:-0}" in
        0|none|n)
            bump_level=""
            ;;
        1|patch|p)
            bump_level="patch"
            ;;
        2|minor)
            bump_level="minor"
            ;;
        3|major)
            bump_level="major"
            ;;
        *)
            echo "Error: select 0, 1, 2, or 3." >&2
            exit 2
            ;;
    esac
fi

echo "Running tests before publishing..."
# This small shim avoids a known native-readline crash in the local Miniconda
# interpreter while leaving the package tests themselves unchanged.
"$python" -c 'import sys, types; sys.modules["readline"] = types.ModuleType("readline"); import pytest; raise SystemExit(pytest.main(["-q"]))'

current_version=""
next_version=""

if [[ -n "$bump_level" ]]; then
    version_info="$("$python" - "$repo_root/pyproject.toml" "$bump_level" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
level = sys.argv[2]
text = path.read_text(encoding="utf-8")
match = re.search(
    r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"\s*$',
    text,
    flags=re.MULTILINE,
)
if match is None:
    raise SystemExit("Could not find a numeric X.Y.Z version in pyproject.toml")

major, minor, patch = map(int, match.groups())
current = f"{major}.{minor}.{patch}"

if level == "major":
    major, minor, patch = major + 1, 0, 0
elif level == "minor":
    minor, patch = minor + 1, 0
else:
    patch += 1

print(current, f"{major}.{minor}.{patch}")
PY
)"
    read -r current_version next_version <<< "$version_info"
fi

if [[ -z "$(git status --porcelain)" && -z "$bump_level" ]]; then
    echo "Nothing to publish: the working tree is clean."
    exit 0
fi

echo
echo "Changes that will be published:"
if [[ -n "$(git status --porcelain)" ]]; then
    git status --short
else
    echo "  (version bump only)"
fi

if [[ -n "$bump_level" ]]; then
    echo "Version bump: $current_version -> $next_version ($bump_level)"
fi

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

if [[ -n "$bump_level" ]]; then
    "$python" - "$repo_root/pyproject.toml" "$current_version" "$next_version" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
old_version = sys.argv[2]
new_version = sys.argv[3]
text = path.read_text(encoding="utf-8")
old_line = f'version = "{old_version}"'

if text.count(old_line) != 1:
    raise SystemExit(f"Expected exactly one {old_line!r} entry")

path.write_text(
    text.replace(old_line, f'version = "{new_version}"', 1),
    encoding="utf-8",
)
PY
fi

git diff --check
git add --all
git commit -m "$commit_message"
git pull --rebase origin "$branch"
git push --set-upstream origin "$branch"

if [[ -n "$bump_level" ]]; then
    echo "Published maioutils $next_version successfully to origin/$branch."
else
    echo "Published successfully to origin/$branch."
fi
