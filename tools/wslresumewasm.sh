#!/bin/sh
# A session saved by the native build, resumed in WebAssembly, under node.
#
# The state was written where a pointer is eight bytes and is read where it is
# four.  If anything in the file were a host address, or sized by a host word,
# this is where it would show - which is what the guest-space arena was for.
set -e
cd "$(dirname "$0")/.."
NODE=$(ls "$HOME"/emsdk/node/*/bin/node 2>/dev/null | head -1)
[ -x "$NODE" ] || { echo "no node in emsdk - run tools/wslgetemsdk.sh"; exit 1; }
exec "$NODE" web/test_resume.mjs "$@"
