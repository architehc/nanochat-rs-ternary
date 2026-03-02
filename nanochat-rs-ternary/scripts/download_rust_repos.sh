#!/bin/bash
# Download popular Rust repos for training data
set -euo pipefail

REPO_DIR="data/rust-repos"
mkdir -p "$REPO_DIR"

# Top Rust repos by quality/size (depth 1 for speed)
REPOS=(
    # Core ecosystem
    "rust-lang/rust"
    "rust-lang/cargo"
    "rust-lang/rustfmt"
    "rust-lang/rust-clippy"
    "rust-analyzer/rust-analyzer"

    # Web/async
    "tokio-rs/tokio"
    "tokio-rs/axum"
    "tokio-rs/mio"
    "tokio-rs/tracing"
    "hyperium/hyper"
    "seanmonstar/reqwest"
    "seanmonstar/warp"
    "actix/actix-web"
    "poem-web/poem"

    # Serialization
    "serde-rs/serde"
    "serde-rs/json"
    "dtolnay/syn"
    "dtolnay/quote"
    "dtolnay/proc-macro2"
    "dtolnay/anyhow"
    "dtolnay/thiserror"

    # CLI/tools
    "clap-rs/clap"
    "BurntSushi/ripgrep"
    "sharkdp/bat"
    "sharkdp/fd"
    "ogham/exa"
    "nushell/nushell"
    "starship/starship"
    "ajeetdsouza/zoxide"
    "alacritty/alacritty"
    "zellij-org/zellij"

    # Database/storage
    "diesel-rs/diesel"
    "launchbadge/sqlx"
    "tikv/tikv"
    "spacejam/sled"

    # Game/graphics
    "bevyengine/bevy"
    "gfx-rs/wgpu"
    "nannou-org/nannou"

    # Crypto/security
    "RustCrypto/hashes"
    "rustls/rustls"

    # Data structures/algorithms
    "rayon-rs/rayon"
    "crossbeam-rs/crossbeam"
    "rust-itertools/itertools"
    "rust-random/rand"
    "chronotope/chrono"
    "regex-rs/regex"
    "BurntSushi/memchr"

    # Embedded/systems
    "rust-embedded/rust-raspberrypi-OS-tutorials"
    "embassy-rs/embassy"

    # Misc popular
    "denoland/deno"
    "nickel-org/nickel.rs"
    "servo/servo"
    "SergioBenitez/Rocket"
    "rust-lang/mdBook"
    "rust-lang/book"
)

echo "Downloading ${#REPOS[@]} Rust repos..."

for repo in "${REPOS[@]}"; do
    name=$(basename "$repo")
    if [ -d "$REPO_DIR/$name" ]; then
        echo "  [skip] $name (exists)"
        continue
    fi
    echo "  [clone] $repo -> $name"
    git clone --depth 1 --quiet "https://github.com/$repo.git" "$REPO_DIR/$name" 2>/dev/null || {
        echo "  [FAIL] $repo"
        continue
    }
done

echo ""
echo "Done. Extracting .rs files..."

# Count total .rs files and size
total_files=$(find "$REPO_DIR" -name "*.rs" -not -path "*/target/*" -not -path "*/.git/*" | wc -l)
total_size=$(find "$REPO_DIR" -name "*.rs" -not -path "*/target/*" -not -path "*/.git/*" -exec du -cb {} + | tail -1 | awk '{print $1}')
echo "Total .rs files: $total_files"
echo "Total size: $((total_size / 1024 / 1024)) MB"
