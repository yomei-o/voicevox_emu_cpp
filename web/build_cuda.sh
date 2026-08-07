#!/bin/sh
# The WebAssembly build with the CUDA shim behind it.
#
# Same emulator as web/build.sh, plus src/cudahost.cpp and the arithmetic it
# calls - so a guest running the CUDA build of voicevox_onnxruntime has its
# kernels answered as compiled WebAssembly instead of as interpreted x86.
#
# A separate module (web/x86emu_cuda.js), not a replacement: the demo page runs
# the ordinary one, and the CUDA path needs a payload the page does not carry.
#
# Needs emscripten - tools/wslgetemsdk.sh fetches one.
set -e
cd "$(dirname "$0")/.."
EMCC=${EMCC:-emcc}
command -v "$EMCC" >/dev/null 2>&1 || [ -x "$EMCC" ] ||
    { echo "emcc not found; set EMCC=/path/to/emcc"; exit 1; }

# The link has to be a C++ one.  emcc decides that from its inputs, and one
# object compiled from C among them is enough for it to decide wrong - the C++
# runtime then goes missing and every `throw` in the emulator comes back as an
# undefined symbol.
#
# The C++ driver is em++, not emcc++ - `${EMCC}++` looks right and names a file
# that does not exist.  The fallback covers it either way.
EMXX=${EMXX:-$(dirname "$EMCC")/em++}
[ -x "$EMXX" ] || command -v "$EMXX" >/dev/null 2>&1 || EMXX="$EMCC -sDEFAULT_TO_CXX"

echo "== building web/x86emu_cuda.js"
SOURCES=$(ls x86_emu_cpp/src/*.cpp | grep -v '/main\.cpp$')
SHIM="src/cudahost.cpp src/cudnn_real.cpp src/cublas_real.cpp"

# src/cudakernels.c is C, and emcc would put -std=c++17 on it along with
# everything else, which clang refuses outright.  Compiled on its own first.
mkdir -p build/wasm
"$EMCC" -O3 -msimd128 -Isrc -c src/cudakernels.c -o build/wasm/cudakernels.o

# -msimd128 for the shim's sake: the convolutions are 90 % of its time and
# WebAssembly's SIMD is what Eigen has to work with here.  The emulator itself
# is built the same way for one binary's worth of consistency.
#
# The heap has to hold the 460 MB provider in MEMFS *and* the guest's own pages,
# which the reserve-don't-allocate change brought down to about 540 MB.  Whether
# that fits is the open question this build exists to answer.
$EMXX -std=c++17 -O3 -msimd128 -DVVCUDA_SHIM \
    -Ix86_emu_cpp/src -Isrc -Ithird_party/eigen_flat \
    $SOURCES $SHIM build/wasm/cudakernels.o web/wasm_api.cpp \
    -o web/x86emu_cuda.js \
    -sMODULARIZE=1 \
    -sEXPORT_NAME=createX86EmuCuda \
    -sALLOW_MEMORY_GROWTH=1 \
    -sMAXIMUM_MEMORY=4GB \
    -sSTACK_SIZE=4MB \
    -sEXPORTED_FUNCTIONS='["_emu_run","_emu_run_path","_emu_resume_path","_emu_set_sysroot","_emu_setenv","_emu_error","_emu_format","_emu_instructions","_malloc","_free"]' \
    -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8","FS","ENV"]' \
    -sFORCE_FILESYSTEM=1 \
    -sDISABLE_EXCEPTION_CATCHING=0 \
    -sENVIRONMENT=node,web,worker \
    --no-entry

ls -l web/x86emu_cuda.js web/x86emu_cuda.wasm
echo done
