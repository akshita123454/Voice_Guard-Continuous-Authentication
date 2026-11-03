#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="data/musan"
ARCHIVE="musan.tar.gz"
URL="https://www.openslr.org/resources/17/musan.tar.gz"
EXPECTED_MD5="0c472d4fc0c51414571f1624b5fa77ee"

echo "================================================================="
echo "VOICEGUARD DATA ACQUISITION: MUSAN CORPUS (OpenSLR 17)"
echo "================================================================="

mkdir -p "${DEST_DIR}"
mkdir -p "data/downloads"
cd "data/downloads"

if [ ! -f "${ARCHIVE}" ]; then
    echo "[1/3] Downloading MUSAN archive (~11 GB)..."
    wget -c --show-progress "${URL}" -O "${ARCHIVE}"
fi

echo "[2/3] Verifying MD5 checksum..."
ACTUAL_MD5=$(md5sum "${ARCHIVE}" | awk '{print $1}')
if [ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]; then
    echo "Checksum mismatch! Expected ${EXPECTED_MD5}, got ${ACTUAL_MD5}"
    exit 1
fi
echo "✓ Checksum OK."

echo "[3/3] Extracting into data/..."
cd ../..
tar -xzf "data/downloads/${ARCHIVE}" -C "data/"

if [ -d "data/musan/musan" ]; then
    mv data/musan/musan/* data/musan/
    rmdir data/musan/musan
fi

echo "✅ MUSAN extraction complete in data/musan"