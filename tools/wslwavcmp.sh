#!/bin/sh
# Two WAVs, compared, with node from emsdk.
#
#   sh tools/wslwavcmp.sh a.wav b.wav
#
# A script rather than a command line because a command substitution does not
# survive `wsl.exe -- bash -c`.
set -e
cd "$(dirname "$0")/.."
NODE=$(ls "$HOME"/emsdk/node/*/bin/node 2>/dev/null | head -1)
[ -x "$NODE" ] || { echo "no node in emsdk"; exit 1; }
exec "$NODE" tools/wavcmp.mjs "$@"
