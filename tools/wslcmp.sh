#!/bin/sh
# Compare what the emulated run just produced against the Tesla T4's own output.
cd "$(dirname "$0")/.."
python3 tools/wavcmp.py sysroot/opt/vvcuda/out.wav "${1:-web/sample/gpu_a.wav}"
