#!/bin/bash
# Build SIFT 1B as 10 sequential segments of 100M docs each.
# Stops early if disk free < 1 TB. Resumes any partial segment via
# phase1.progress + atomic .srt writes (no work lost on crash/interrupt).

set -e
BASE=/home/chatelet/mangrove-search/bigann_base.bvecs
ROOT=/mnt/mangrove/indexes/sift1b
RPFOREST=/home/chatelet/mangrove-search/rpforest
N_TREES=1000
DEPTH=30
SUB_DIM=16
GEN=v3
DIM=128

mkdir -p "$ROOT"
echo "Building SIFT 1B as 10 segments × 100M docs each ..."
echo "Target: $ROOT/seg{0..9} | n_trees=$N_TREES depth=$DEPTH sub_dim=$SUB_DIM gen=$GEN"

for i in 0 1 2 3 4 5 6 7 8 9; do
    SEG_DIR=$ROOT/seg$i
    OFFSET=$((i * 100000000))
    echo
    echo "=== Segment $i (offset $OFFSET) ==="

    # Check disk: stop if less than 1 TB free
    AVAIL_GB=$(df --output=avail -BG /mnt/mangrove | tail -1 | tr -d 'G ')
    echo "Disk free: ${AVAIL_GB} GB"
    if [ "$AVAIL_GB" -lt 1000 ]; then
        echo "STOPPING: less than 1 TB free."
        break
    fi

    mkdir -p "$SEG_DIR"
    "$RPFOREST" --dim $DIM --sub_dim $SUB_DIM --gen $GEN \
        --doc_offset $OFFSET --doc_count 100000000 \
        build "$BASE" "$SEG_DIR" $N_TREES $DEPTH
    echo "Segment $i built. Verifying ..."
    "$RPFOREST" verify "$SEG_DIR" $N_TREES | tail -2
done

echo
echo "Multi-index 1B build pass complete."
ls -la "$ROOT/"
df -h /mnt/mangrove
