#!/bin/sh
# The longer utterance through the emulated CUDA path, then compared with the
# Tesla T4's own output.  A separate script rather than an argument because a
# Japanese string does not survive the trip from a Windows shell through
# wsl.exe, and the point of this one is that the text is right.
cd "$(dirname "$0")/.."
sh tools/wslrun_cuda.sh "ずんだもんなのだ" 2>&1 | grep -vE "^x86emu: open|^kernel " | tail -8
echo
python3 tools/wavcmp.py sysroot/opt/vvcuda/out.wav web/sample/gpu_zundamon.wav
