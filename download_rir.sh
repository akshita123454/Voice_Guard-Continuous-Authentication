#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="data/rirs_noises"
ARCHIVE="rirs_noises.zip"
URL="https://www.openslr.org/resources/28/rirs_noises.zip"
EXPECTED_MD5="e6f48e257286e05de56413f4f120e8f9"

echo "================================================================="
echo "VOICEGUARD DATA ACQUISITION: RIR & ROOM NOISES (OpenSLR 28)"
echo "================================================================="

mkdir -p "data/downloads"
cd "data/downloads"

if [ ! -f "${ARCHIVE}" ]; then
    echo "[1/3] Downloading RIRs archive (~4 GB)..."
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
unzip -q -u "data/downloads/${ARCHIVE}" -d "data/"

echo "✅ RIR extraction complete in data/RIRS_NOISES"
