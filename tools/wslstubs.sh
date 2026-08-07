#!/bin/sh
# Build the guest-mode stand-ins under WSL, logging where WSL will not lose it.
#
# A wrapper because the trip from a Windows shell through wsl.exe eats variable
# assignments and redirections; a script file does not go through it.
cd "$(dirname "$0")/.."
LOG=$HOME/vvstubs.log
MODE=guest CXX=$HOME/gpp/bin/g++ CC=gcc \
    sh tools/make_cuda_stubs.sh \
    "$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib" > "$LOG" 2>&1
status=$?
cut -c1-220 "$LOG" | head -50
exit $status
