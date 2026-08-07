#!/bin/sh
# The native side of the argument comparison, on its own.
#
# It touches guest/cudastub only, never guest/cudaguest or ./vvcudaemu, so it
# can run while the emulated side is busy under a debugger.
set -e
cd "$(dirname "$0")/.."
mkdir -p build
CXX=$HOME/gpp/bin/g++ CC=gcc MODE=native \
    sh tools/make_cuda_stubs.sh \
    "$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib" > "$HOME/vvnative.log" 2>&1 ||
    { tail -20 "$HOME/vvnative.log"; exit 1; }
cp guest/cudastub/*.so.* "$HOME/vv/cudarun/"
( cd "$HOME/vv/cudarun" &&
  VVSTUB_KARGS=1 LD_LIBRARY_PATH=. ./cudavvm ./libvoicevox_onnxruntime.so.1.17.3 \
      ./open_jtalk_dic_utf_8-1.11 ./0.vvm 3 "ずんだもんなのだ" /tmp/n.wav 2>/dev/null ) |
  grep -E ', [0-9]+ args$|^ *\[[0-9]+\]' | head -60 > build/a_native.txt || true
cat build/a_native.txt
