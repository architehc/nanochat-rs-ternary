#!/bin/bash
# Download top Rust repositories for training data

set -eo pipefail

DATA_DIR="data/rust-repos"
mkdir -p "$DATA_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  Downloading Top Rust Repositories"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Top Rust crates by downloads (excluding already downloaded)
REPOS=(
    "https://github.com/rust-lang/rust.git rust-lang"           # The Rust compiler
    "https://github.com/rust-lang/cargo.git cargo"              # Package manager
    "https://github.com/actix/actix-web.git actix-web"          # Web framework
    "https://github.com/hyperium/hyper.git hyper"               # HTTP library
    "https://github.com/servo/servo.git servo"                  # Browser engine
    "https://github.com/rust-lang/regex.git regex"              # Regex
    "https://github.com/rayon-rs/rayon.git rayon"               # Parallel computing
    "https://github.com/diesel-rs/diesel.git diesel"            # ORM
    "https://github.com/crossbeam-rs/crossbeam.git crossbeam"   # Concurrency
    "https://github.com/tikv/tikv.git tikv"                     # Distributed KV
)

for repo_info in "${REPOS[@]}"; do
    url=$(echo "$repo_info" | awk '{print $1}')
    name=$(echo "$repo_info" | awk '{print $2}')

    echo "─────────────────────────────────────────────────────────"
    echo "Repository: $name"
    echo "URL: $url"

    if [ -d "$DATA_DIR/$name" ]; then
        echo "✓ Already exists, skipping"
        continue
    fi

    echo "Cloning (shallow, depth=1 for speed)..."
    if git clone --depth 1 "$url" "$DATA_DIR/$name"; then
        echo "✓ Cloned successfully"

        # Count Rust files
        rust_files=$(find "$DATA_DIR/$name" -name "*.rs" 2>/dev/null | wc -l)
        size=$(du -sh "$DATA_DIR/$name" 2>/dev/null | cut -f1)
        echo "  Rust files: $rust_files"
        echo "  Size: $size"
    else
        echo "✗ Failed to clone (might be too large, skipping)"
        rm -rf "$DATA_DIR/$name"
    fi

    echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo "  Download Summary"
echo "═══════════════════════════════════════════════════════════"
echo ""

for dir in "$DATA_DIR"/*; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        rust_files=$(find "$dir" -name "*.rs" 2>/dev/null | wc -l)
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        printf "%-20s %6d files  %8s\n" "$name" "$rust_files" "$size"
    fi
done

total_files=$(find "$DATA_DIR" -name "*.rs" 2>/dev/null | wc -l)
total_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
echo ""
echo "Total: $total_files Rust files, $total_size"
echo ""
echo "✓ Download complete!"
