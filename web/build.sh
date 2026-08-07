#!/bin/sh
# Builds the browser demo: the emulator, compiled to WebAssembly.
#
# The wasm is embedded in x86emu.js (SINGLE_FILE), so web/ is plain static files
# - no MIME setup, no separate .wasm fetch, and it works straight from GitHub
# Pages.  The VOICEVOX payload is far too large to embed and is fetched by the
# page at run time; make_payload.sh puts it under web/payload/.
#
# Needs emscripten on PATH, or EMCC pointing at emcc.
set -e
cd "$(dirname "$0")/.."
EMCC=${EMCC:-emcc}
command -v "$EMCC" >/dev/null 2>&1 || { echo "emcc not found; set EMCC=/path/to/emcc"; exit 1; }

echo "== building web/x86emu.js"
# Everything in the emulator's src/ except the command line front end, which
# web/wasm_api.cpp replaces.  Globbing keeps this from going stale.
SOURCES=$(ls x86_emu_cpp/src/*.cpp | grep -v '/main\.cpp$')

# A 58 MB model, a 100 MB dictionary and an 18 MB runtime all live in the guest
# at once, so the heap has to be allowed to grow well past the default.
"$EMCC" -std=c++17 -O3 -Ix86_emu_cpp/src \
    $SOURCES web/wasm_api.cpp \
    -o web/x86emu.js \
    -sMODULARIZE=1 \
    -sEXPORT_NAME=createX86Emu \
    -sSINGLE_FILE=1 \
    -sALLOW_MEMORY_GROWTH=1 \
    -sMAXIMUM_MEMORY=4GB \
    -sSTACK_SIZE=4MB \
    -sEXPORTED_FUNCTIONS='["_emu_run","_emu_run_path","_emu_set_sysroot","_emu_setenv","_emu_guest_setenv","_emu_error","_emu_format","_emu_instructions","_malloc","_free"]' \
    -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8","FS"]' \
    -sFORCE_FILESYSTEM=1 \
    -sDISABLE_EXCEPTION_CATCHING=0 \
    -sENVIRONMENT=web,worker,node \
    --no-entry

ls -l web/x86emu.js
echo "done - serve web/ over http and open index.html"
