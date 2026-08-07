#!/bin/sh
# cudaprobe under vvcudaemu: the same shim path as a full run, in seconds.
#
# The full one spends fourteen minutes building sessions before it launches a
# kernel, which is a poor loop to debug in.  This loads one small plain .onnx
# and runs it - eight launches, and the sixteen values that have to match.
set -e
cd "$(dirname "$0")/.."
OUT=sysroot/opt/vvcuda
CUDA=${VVCUDA:-$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib}

cp guest/cudaprobe "$OUT/" 2>/dev/null || cp "$HOME/vv/cudarun/cudaprobe" "$OUT/"
cp guest/predict_duration.onnx "$OUT/"
mkdir -p sysroot/lib/x86_64-linux-gnu
cp guest/cudaguest/*.so.* sysroot/lib/x86_64-linux-gnu/

./vvcudaemu --sysroot "$PWD/sysroot" "$OUT/cudaprobe" \
    /opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 /opt/vvcuda/predict_duration.onnx 2>&1 |
    grep -vE "^x86emu: open" | tail -30
