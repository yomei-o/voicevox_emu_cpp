#!/bin/sh
# The native shim, after the split into cudainfo.c: still exact, still audible.
#
# The guest build and the native build now share src/cudainfo.c, so a change for
# one can break the other silently.  This is the check that says it did not.
cd "$(dirname "$0")/.."
CXX=$HOME/gpp/bin/g++ CC=gcc MODE=native \
    sh tools/make_cuda_stubs.sh \
    "$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib" > "$HOME/vvnative.log" 2>&1 ||
    { tail -20 "$HOME/vvnative.log"; exit 1; }
tail -2 "$HOME/vvnative.log"

cp guest/cudastub/*.so.* "$HOME/vv/cudarun/"
sh tools/check_shim.sh | tail -2
cd "$HOME/vv/cudarun"
LD_LIBRARY_PATH=. ./cudavvm ./libvoicevox_onnxruntime.so.1.17.3 \
    ./open_jtalk_dic_utf_8-1.11 ./0.vvm 3 "ずんだもんなのだ" native.wav 2>/dev/null |
    grep -E "tts took|produced"
cd - > /dev/null
python3 tools/wavcmp.py "$HOME/vv/cudarun/native.wav" web/sample/gpu_zundamon.wav | tail -3
