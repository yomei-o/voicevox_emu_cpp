#!/bin/sh
# A named stack for the WebAssembly crash, and a loop that takes seconds.
#
# "table index is out of bounds at wasm-function[627]" says nothing.  Built with
# --profiling-funcs the frames have names, and -sASSERTIONS turns several common
# mistakes into sentences.  cudaprobe reaches the same kernel-launch path as the
# full run without the fourteen minutes of session building in front of it.
set -e
cd "$(dirname "$0")/.."
EMCC=$HOME/emsdk/upstream/emscripten/emcc
EMXX=$HOME/emsdk/upstream/emscripten/em++
NODE=$(ls "$HOME"/emsdk/node/*/bin/node | head -1)

mkdir -p build/wasm
SOURCES=$(ls x86_emu_cpp/src/*.cpp | grep -v '/main\.cpp$')
SHIM="src/cudahost.cpp src/cudnn_real.cpp src/cublas_real.cpp"

# SIMD=0 builds without -msimd128.  Eigen picks its packet maths from the
# target, so this is the one switch that separates "the shim is wrong" from
# "the shim is wrong *vectorised*" - and the answers differ between -O1 and -O3,
# which is what makes that worth asking.
SIMD=${SIMD:--msimd128}
[ "$SIMD" = "0" ] && SIMD=""
echo "== simd: ${SIMD:-none}"

"$EMCC" -O1 -g2 $SIMD -Isrc -c src/cudakernels.c -o build/wasm/cudakernels_dbg.o

echo "== building web/x86emu_cuda_dbg.js"
"$EMXX" -std=c++17 -O1 -g2 --profiling-funcs $SIMD -DVVCUDA_SHIM \
    -sASSERTIONS=2 -sSAFE_HEAP=0 \
    -Ix86_emu_cpp/src -Isrc -Ithird_party/eigen_flat \
    $SOURCES $SHIM build/wasm/cudakernels_dbg.o web/wasm_api.cpp \
    -o web/x86emu_cuda_dbg.js \
    -sMODULARIZE=1 -sEXPORT_NAME=createX86EmuCuda \
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=8MB \
    -sEXPORTED_FUNCTIONS='["_emu_run","_emu_run_path","_emu_set_sysroot","_emu_setenv","_emu_error","_emu_format","_emu_instructions","_malloc","_free"]' \
    -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8","FS","ENV"]' \
    -sFORCE_FILESYSTEM=1 -fwasm-exceptions \
    -sENVIRONMENT=node,web,worker --no-entry

echo "== running"
VVMODULE=$PWD/web/x86emu_cuda_dbg.js "$NODE" web/test_cuda_probe.mjs 2>&1 | tail -40
