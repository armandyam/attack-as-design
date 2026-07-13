#!/bin/bash
set -e
export HF_HUB_DISABLE_XET=1
if [ ! -d /work/evo2 ]; then git clone --depth 1 https://github.com/arcinstitute/evo2 /work/evo2; fi
pip install -q -e /work/evo2 2>&1 | tail -2
python3 probe_evo2.py
