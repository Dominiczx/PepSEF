#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

message=${1:-"update PepSEF code"}

echo "Repository: $repo_root"
echo "Commit message: $message"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This directory is not a Git repository. Run: git init -b main"
  exit 1
fi

echo
echo "Files that will be considered:"
git status --short

git add -A

if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  git commit -m "$message"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git push
else
  echo "No origin remote configured. Add one with:"
  echo "  git remote add origin <your-github-repo-url>"
  echo "  git push -u origin main"
fi

