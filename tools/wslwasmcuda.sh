#!/bin/sh
# Build the CUDA-shim WebAssembly module and measure it under node.
#
# The browser number for this path has only ever been a calculation.  This is
# the measurement.
set -e
cd "$(dirname "$0")/.."
# emsdk_env.sh is written for bash and does nothing useful under dash, so the
# tools are named outright.
EMCC=$HOME/emsdk/upstream/emscripten/emcc
NODE=$(ls "$HOME"/emsdk/node/*/bin/node | head -1)
export EMCC
[ -x "$EMCC" ] || { echo "no emcc at $EMCC - run tools/wslgetemsdk.sh"; exit 1; }

echo "== the guest stand-ins must be the forwarding ones, and SSE2"
[ -f guest/cudaguest/libcudart.so.12 ] ||
    MODE=guest CXX=$HOME/gpp/bin/g++ CC=gcc sh tools/make_cuda_stubs.sh \
        "$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib" > /dev/null

echo "== the slimmed provider, if it is not already there"
SRC=$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib
[ -f "$SRC/libvoicevox_onnxruntime_providers_cuda_slim.so" ] ||
    sh tools/slim_provider.sh "$SRC/libvoicevox_onnxruntime_providers_cuda.so" \
        "$SRC/libvoicevox_onnxruntime_providers_cuda_slim.so" > /dev/null

sh unpack.sh > /dev/null 2>&1 || true
EMCC="$EMCC" sh web/build_cuda.sh
echo
echo "== running under node"
"$NODE" web/test_cuda.mjs "${1:-あ}"
