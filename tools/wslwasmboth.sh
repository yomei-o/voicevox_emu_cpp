#!/bin/sh
# Both WebAssembly modules, because both are committed.
#
# web/x86emu.js and web/x86emu_cuda.js are served by GitHub Pages as they are,
# so a change to web/wasm_api.cpp or to either build script means rebuilding and
# recommitting them - a page running last week's module against this week's
# worker fails in ways that look like nothing in the source.
set -e
cd "$(dirname "$0")/.."
EMCC=${EMCC:-$HOME/emsdk/upstream/emscripten/emcc}
[ -x "$EMCC" ] || { echo "no emcc at $EMCC - run tools/wslgetemsdk.sh"; exit 1; }
export EMCC

echo "== the ordinary module"
sh web/build.sh > /dev/null
echo "== the CUDA module"
sh web/build_cuda.sh > /dev/null
ls -l web/x86emu.js web/x86emu.wasm web/x86emu_cuda.js web/x86emu_cuda.wasm 2>/dev/null
