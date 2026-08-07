#!/bin/sh
# Rebuild vvcudaemu and run the longer utterance with the boundary timed.
cd "$(dirname "$0")/.."
sh tools/wslbuild.sh > /dev/null 2>&1 || { sh tools/wslbuild.sh; exit 1; }
VVSTUB_TIME=1 sh tools/wslrun_cuda.sh "ずんだもんなのだ" 2>&1 |
    grep -vE "^x86emu: open|^kernel " | tail -12
python3 tools/wavcmp.py sysroot/opt/vvcuda/out.wav web/sample/gpu_zundamon.wav | tail -3
