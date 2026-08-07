#!/bin/sh
# Run the CUDA build under vvcudaemu - the emulator with the shim behind it.
#
# Everything Linux-side: WSL runs the emulator, the guest, and the host shim in
# one process, which avoids the whole class of trouble that came from a Windows
# emulator reading a POSIX sysroot (paths rewritten, environment variables
# converted, arguments re-encoded).
#
#   sh tools/wslrun_cuda.sh ["text"]
set -e
cd "$(dirname "$0")/.."
REPO=$PWD
CUDA=${VVCUDA:-$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib}
OUT=sysroot/opt/vvcuda
TEXT=${1:-あ}

[ -x ./vvcudaemu ] || { echo "no ./vvcudaemu - run tools/wslbuild.sh"; exit 1; }

sh unpack.sh
mkdir -p "$OUT" sysroot/lib/x86_64-linux-gnu
cp guest/cudavvm guest/libvoicevox_core.so guest/0.vvm "$OUT/"
cp guest/cudaguest/*.so.* sysroot/lib/x86_64-linux-gnu/
cp "$CUDA/libvoicevox_onnxruntime.so.1.17.3" "$OUT/"
cp "$CUDA/libvoicevox_onnxruntime_providers_shared.so" "$OUT/"

slim=$HOME/vv/slimtest/libvoicevox_onnxruntime_providers_cuda.so
[ -f "$slim" ] || slim="$CUDA/libvoicevox_onnxruntime_providers_cuda.so"
cp "$slim" "$OUT/libvoicevox_onnxruntime_providers_cuda.so"
echo "provider $(wc -c < "$slim") bytes"

[ -d "$OUT/open_jtalk_dic_utf_8-1.11" ] || cp -r guest/open_jtalk_dic_utf_8-1.11 "$OUT/"
printf '%s' "$TEXT" > "$OUT/text.txt"

# A Linux guest gets only PATH unless it is told otherwise - see --env in
# src/vvcudaemu.cpp - so anything the guest program reads has to be passed here.
env_args=""
[ -n "$VVSNAPSHOT" ] && env_args="--env VVSNAPSHOT=$VVSNAPSHOT"
[ -n "$VVSTUB_TRACE" ] && env_args="$env_args --env VVSTUB_TRACE=$VVSTUB_TRACE"

exec ./vvcudaemu --sysroot "$REPO/sysroot" $env_args "$OUT/cudavvm" \
    /opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 \
    /opt/vvcuda/open_jtalk_dic_utf_8-1.11 \
    /opt/vvcuda/0.vvm 3 @/opt/vvcuda/text.txt /opt/vvcuda/out.wav
