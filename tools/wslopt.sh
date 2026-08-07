#!/bin/sh
# Does ORT's graph optimiser account for the 192 seconds, or not?
#
# Building the decode session is 93 % of a run's setup, and its time is
# proportional to the bytes rather than to the graph.  But "proportional to the
# bytes" covers three jobs at once - decrypting them, parsing them, copying the
# weights - and this separates one from the other two.  Turning the optimiser
# off still decrypts and still parses.
set -e
cd "$(dirname "$0")/.."
OUT=sysroot/opt/vvcuda
"${CXX:-$HOME/gpp/bin/g++}" -O2 -std=c++17 -Isrc -o "$OUT/vvmload" src/vvmload.cpp -ldl

for level in "" 0 99; do
    echo "== optimization level ${level:-(the runtime's default)}"
    ./vvcudaemu --sysroot "$PWD/sysroot" "$OUT/vvmload" \
        /opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 /opt/vvcuda/0.vvm decode $level 2>&1 |
        grep -vE "^x86emu: open" | grep -E "^ok    session|^step  graph|FAIL"
done
