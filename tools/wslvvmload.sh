#!/bin/sh
# Build and run the .vvm loading sample against the CPU build of the runtime.
set -e
cd "$(dirname "$0")/.."
"${CXX:-$HOME/gpp/bin/g++}" -std=c++17 -O2 -Isrc -o build/vvmload src/vvmload.cpp -ldl
for which in predict_duration predict_intonation decode; do
    echo "== $which"
    ( cd "$HOME/vv/cudarun" &&
      LD_LIBRARY_PATH=. /mnt/c/prog/claude/voicevox_emu_cpp/build/vvmload \
          ./libvoicevox_onnxruntime.so.1.17.3 ./0.vvm "$which" ) || true
    echo
done
