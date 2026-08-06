#!/bin/sh
# What the shim's arithmetic costs, against each set of optimiser flags.
#
# The CUDA provider hands *all* of its arithmetic across the shim boundary, so
# the three buckets VVSTUB_TIME prints are the arithmetic and whatever the
# caller measured beyond them is ONNX Runtime's own plumbing.  On this model
# that split is about 99 to 1, which is the number that decides whether moving
# the arithmetic out of an emulator is worth the work.
#
#   sh tools/bench_shim.sh <cuda lib dir> [run dir]
set -e
cd "$(dirname "$0")/.."
REPO=$PWD
SRC=${1:-$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib}
RUN=${2:-$HOME/vv/cudarun}
TEXT=${TEXT:-ずんだもんなのだ}

printf '%-34s %10s %10s %10s\n' 'flags' 'cuDNN' 'kernels' 'tts'
for opt in "-O2" "-O3" "-O3 -mavx2 -mfma" "-O3 -march=native"; do
    ( cd "$REPO" && OPT="$opt" sh tools/make_cuda_stubs.sh "$SRC" >/dev/null 2>&1 )
    cp "$REPO/guest/cudastub/"*.so.* "$RUN/"
    out=$( cd "$RUN" && VVSTUB_TIME=1 LD_LIBRARY_PATH=. ./cudavvm \
        ./libvoicevox_onnxruntime.so.1.17.3 ./open_jtalk_dic_utf_8-1.11 ./0.vvm 3 \
        "$TEXT" /tmp/bench.wav 2>&1 )
    cudnn=$(echo "$out" | sed -n 's/.*cuDNN *\([0-9.]*\) s.*/\1/p')
    kern=$(echo "$out" | sed -n 's/.*kernels *\([0-9.]*\) s.*/\1/p')
    tts=$(echo "$out" | sed -n 's/.*tts took \([0-9.]*\) s.*/\1/p')
    printf '%-34s %10s %10s %10s\n' "$opt" "$cudnn" "$kern" "$tts"
done
