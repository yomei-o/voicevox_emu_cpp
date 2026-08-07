#!/bin/sh
# What the demo page does, without a browser: build, save, resume twice.
#
# Checking the page by hand costs twenty minutes of watching a progress bar and
# proves it once.  This runs the same sequence through the same entry points, so
# it can be run after a change rather than before a release.
set -e
cd "$(dirname "$0")/.."
NODE=$(ls "$HOME"/emsdk/node/*/bin/node 2>/dev/null | head -1)
[ -x "$NODE" ] || { echo "no node in emsdk - run tools/wslgetemsdk.sh"; exit 1; }
exec "$NODE" web/test_roundtrip.mjs "$@"
