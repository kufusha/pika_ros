#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION="${VERSION:-$(git -C "$REPO_DIR" rev-parse --short HEAD)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/dist}"
ARCHIVE="$OUTPUT_DIR/pika_ros-g1-pika-retarget-$VERSION.tar.gz"

mkdir -p "$OUTPUT_DIR"
git -C "$REPO_DIR" archive \
  --format=tar.gz \
  --prefix="pika_ros-g1-pika-retarget-$VERSION/" \
  --output="$ARCHIVE" \
  HEAD
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"

echo "archive : $ARCHIVE"
echo "sha256 : $ARCHIVE.sha256"
