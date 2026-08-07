#!/bin/sh
# What carrying the CUDA runtime would cost this repository.
#
# The slimmed provider is still 460 MB on disk - slimming replaces the GPU code
# with zeros rather than removing it - so what matters is what it compresses to,
# which is what git stores and what a page would fetch.
set -e
CUDA=${VVCUDA:-$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib}
SLIM=$HOME/vv/slimtest/libvoicevox_onnxruntime_providers_cuda.so
[ -f "$SLIM" ] || SLIM=$CUDA/libvoicevox_onnxruntime_providers_cuda_slim.so

for f in "$CUDA/libvoicevox_onnxruntime.so.1.17.3" \
         "$CUDA/libvoicevox_onnxruntime_providers_shared.so" \
         "$SLIM"; do
    [ -f "$f" ] || { echo "missing: $f"; continue; }
    raw=$(stat -c%s "$f")
    gz=$(gzip -9 -c "$f" | wc -c)
    printf '%-46s raw %8.1f MB   gz %6.1f MB\n' "$(basename "$f")" \
        "$(awk "BEGIN{print $raw/1048576}")" "$(awk "BEGIN{print $gz/1048576}")"
done
