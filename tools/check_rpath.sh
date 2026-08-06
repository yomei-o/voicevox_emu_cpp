#!/bin/sh
# Does anything here carry an RPATH?  It matters because RPATH beats
# LD_LIBRARY_PATH: a library that names its own search path cannot be pointed
# somewhere else by the environment, which is a confusing thing to discover
# while trying to substitute a CUDA runtime.
#
#   sh tools/check_rpath.sh <dir>
set -e
DIR=${1:-.}
for f in "$DIR"/*.so*; do
    [ -f "$f" ] || continue
    line=$(readelf -d "$f" 2>/dev/null | grep -E 'RPATH|RUNPATH' | head -1)
    printf '%-52s %s\n' "$(basename "$f")" "${line:-(no rpath)}"
done
