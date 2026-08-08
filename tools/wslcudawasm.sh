#!/bin/sh
# The CUDA path in WebAssembly, measured, with node from emsdk.
#
# A script rather than a command line because a command substitution does not
# survive `wsl.exe -- bash -c`: it arrives empty, and the run dies with
# `: command not found` before it has measured anything.
set -e
cd "$(dirname "$0")/.."
NODE=$(ls "$HOME"/emsdk/node/*/bin/node 2>/dev/null | head -1)
[ -x "$NODE" ] || { echo "no node in emsdk"; exit 1; }
exec "$NODE" web/test_cuda.mjs "$@"
