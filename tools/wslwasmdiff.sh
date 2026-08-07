#!/bin/sh
# The same eight kernels, native and WebAssembly, side by side.
#
# The WebAssembly build answers predict_duration with seven values of garbage
# and nine that are plausible and wrong.  VVSTUB_STATS prints the range of what
# each kernel wrote, and the first line that differs is where to look - which is
# how the shift-of-six and the unary Div were found.
set -e
cd "$(dirname "$0")/.."
NODE=$(ls "$HOME"/emsdk/node/*/bin/node | head -1)

echo "== native"
( cd "$HOME/vv/cudarun" &&
  VVSTUB_STATS=1 LD_LIBRARY_PATH=. ./cudaprobe ./libvoicevox_onnxruntime.so.1.17.3 \
      ./predict_duration.onnx 2>&1 >/dev/null | grep '^\[k\]' ) > build/stats_native.txt || true
cat build/stats_native.txt

echo
echo "== webassembly"
VVSTUB_STATS=1 VVMODULE=$PWD/web/x86emu_cuda_dbg.js "$NODE" web/test_cuda_probe.mjs 2>&1 |
    grep '^\[k\]' > build/stats_wasm.txt || true
cat build/stats_wasm.txt

echo
echo "== first difference"
diff build/stats_native.txt build/stats_wasm.txt | head -12 || true
