#!/bin/sh
# Put the CUDA runtime into the repository, gzipped.
#
# Without it the demo page cannot start until someone has downloaded a 440 MB
# release and picked three files out of it, which is a poor first impression for
# a page whose whole point is that it runs the CUDA build without a GPU.
#
# Gzipped because the slimmed provider is *still* 439 MB on disk - slimming
# replaces the GPU machine code with zeros rather than removing it - and a clone
# should not write 439 MB of nothing.  It compresses to 8.8 MB, and the page
# already knows how to gunzip: the Open JTalk dictionary arrives the same way.
set -e
cd "$(dirname "$0")/.."
CUDA=${VVCUDA:-$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib}
SLIM=$HOME/vv/slimtest/libvoicevox_onnxruntime_providers_cuda.so
[ -f "$SLIM" ] || SLIM=$CUDA/libvoicevox_onnxruntime_providers_cuda_slim.so
[ -f "$SLIM" ] || { echo "no slimmed provider - run tools/slim_provider.sh first"; exit 1; }

mkdir -p guest/cuda
gzip -9 -c "$CUDA/libvoicevox_onnxruntime.so.1.17.3" \
    > guest/cuda/libvoicevox_onnxruntime.so.1.17.3.gz
gzip -9 -c "$CUDA/libvoicevox_onnxruntime_providers_shared.so" \
    > guest/cuda/libvoicevox_onnxruntime_providers_shared.so.gz
gzip -9 -c "$SLIM" \
    > guest/cuda/libvoicevox_onnxruntime_providers_cuda.so.gz
ls -l guest/cuda/
echo
echo "the provider carried here is the slimmed one: its GPU code is zeros, which"
echo "is why 439 MB becomes 8.8 MB and why it behaves identically - none of that"
echo "code is ever executed on this path."
