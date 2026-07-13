#!/bin/bash
set -e
echo "=== installing evo2 (NGC already has transformer_engine + flash-attn) ==="
export HF_HUB_DISABLE_XET=1
if [ ! -d /work/evo2 ]; then
  git clone --depth 1 https://github.com/arcinstitute/evo2 /work/evo2
fi
pip install -q -e /work/evo2 2>&1 | tail -3
echo "=== running extraction ==="
python3 extract_evo2_8mers.py
