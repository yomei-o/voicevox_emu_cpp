#!/bin/sh
# Put the saved session where Windows can see it.
#
# It lives in the WSL home by default, which Explorer only reaches through
# \\wsl.localhost.  build/ is ignored by git, and *.state with it, so a 166 MB
# file of decrypted weights cannot be committed by accident from either rule.
set -e
cd "$(dirname "$0")/.."
SNAP=${VVSNAP:-$HOME/vv/session.state}
DEST=${1:-$PWD/build}
[ -f "$SNAP" ] || { echo "no $SNAP - run tools/wslresume.sh first"; exit 1; }
mkdir -p "$DEST"
cp -v "$SNAP" "$DEST/session.state"
cp -v "$SNAP.shim" "$DEST/session.state.shim"
ls -l "$DEST"/session.state*
