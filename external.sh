#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
DEST="$ROOT/external"
GITMODULES="$ROOT/.gitmodules"

# Format: name|repo_url|tag|sparse_paths (space-separated)
deps=(
  "xtl|https://github.com/xtensor-stack/xtl.git|0.8.0|include/ LICENSE"
  "xtensor|https://github.com/xtensor-stack/xtensor.git|0.27.0|include/ LICENSE"
  "xtensor-blas|https://github.com/xtensor-stack/xtensor-blas.git|0.22.0|include/ LICENSE"
)

mkdir -p "$DEST"

[ -f "$GITMODULES" ] || { echo "# Created automatically" > "$GITMODULES"; git add "$GITMODULES"; }

remove_submodule() {
  local path="$1"
  if git config -f "$GITMODULES" --get-regexp "submodule.$path.path" >/dev/null 2>&1; then
    echo "Removing stale submodule: $path"
    git submodule deinit -f "$path" || true
    git rm -f "$path" || true
    rm -rf ".git/modules/$path"
  fi
  rm -rf "$path"
}

# Remove all existing submodules for these deps
for dep in "${deps[@]}"; do
  IFS="|" read -r name _ _ _ <<< "$dep"
  remove_submodule "external/$name"
done
rm -rf .git/modules/external || true

for dep in "${deps[@]}"; do
  IFS="|" read -r name repo_url tag sparse_paths <<< "$dep"

  echo "Adding $name @ $tag"
  git submodule add "$repo_url" "external/$name"

  (
    cd "$DEST/$name"
    git fetch --tags --depth 1 origin tag "$tag"
    git checkout "tags/$tag" --detach

    actual_tag="$(git describe --tags --exact-match 2>/dev/null || true)"
    if [[ "$actual_tag" != "$tag" ]]; then
      echo "ERROR: $name checked out to '$actual_tag', expected '$tag'"
      exit 1
    fi

    git config core.sparseCheckout true
    mkdir -p "$(git rev-parse --git-dir)/info"
    sparse_file="$(git rev-parse --git-dir)/info/sparse-checkout"
    : > "$sparse_file"
    for path in $sparse_paths; do
      echo "$path" >> "$sparse_file"
    done
    git read-tree -mu HEAD
  )
done

echo "✅ Vendored dependencies at locked versions."
