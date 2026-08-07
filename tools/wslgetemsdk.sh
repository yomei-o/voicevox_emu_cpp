#!/bin/sh
# Fetch emscripten into $HOME/emsdk, for building web/ and for the node it
# brings with it.
#
# Neither emcc nor node is present on this machine, and web/x86emu.js in the
# tree is a build from an earlier session - so anything that needs a *new*
# WebAssembly build needs this first.  About 1-2 GB and ten to twenty minutes.
set -e
PREFIX=$HOME/emsdk
if [ -x "$PREFIX/upstream/emscripten/emcc" ]; then
    echo "already at $PREFIX"
else
    command -v git >/dev/null 2>&1 || { echo "no git"; exit 1; }
    command -v python3 >/dev/null 2>&1 || { echo "no python3"; exit 1; }
    [ -d "$PREFIX" ] || git clone --depth 1 https://github.com/emscripten-core/emsdk "$PREFIX"
    cd "$PREFIX"
    ./emsdk install latest
    ./emsdk activate latest
fi
. "$PREFIX/emsdk_env.sh" > /dev/null 2>&1 || true
echo "== versions"
"$PREFIX/upstream/emscripten/emcc" --version | head -1
ls "$PREFIX"/node/*/bin/node 2>/dev/null | head -1
