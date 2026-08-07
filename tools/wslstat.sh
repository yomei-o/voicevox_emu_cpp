#!/bin/sh
# Summarise the last emulated run's output, and the reference beside it.
cd "$(dirname "$0")/.."
python3 tools/wavstat.py sysroot/opt/vvcuda/out.wav
echo
echo "  for comparison, the Tesla T4's own ずんだもんなのだ:"
python3 tools/wavstat.py web/sample/gpu_zundamon.wav
