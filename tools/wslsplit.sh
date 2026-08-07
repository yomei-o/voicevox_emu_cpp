#!/bin/sh
# Decryption or session building - which of the two is the 206 seconds?
#
# `load_voice_model` does both in one call, so they cannot be separated by
# timing it.  But the three models in a .vvm differ by three orders of magnitude
# in size, and the two jobs scale with different things:
#
#   decryption      with the bytes        52 KB : 30 KB : 58 MB
#   session build   with the graph        two small graphs and a vocoder
#
# So timing each of the three separately says which.  If the 58 MB model takes a
# thousand times what the 52 KB one does, it is the bytes; if it takes three
# times, it is the graph.
set -e
cd "$(dirname "$0")/.."
OUT=sysroot/opt/vvcuda

echo "== building vvmload for the guest (SSE2, like everything else in there)"
# The system gcc has no C++ backend here; tools/get_gpp_nosudo.sh unpacks one.
"${CXX:-$HOME/gpp/bin/g++}" -O2 -std=c++17 -Isrc -o "$OUT/vvmload" src/vvmload.cpp -ldl
cp "$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib/libvoicevox_onnxruntime.so.1.17.3" \
   "$OUT/" 2>/dev/null || true

echo
echo "== native, for scale"
for which in predict_duration predict_intonation decode; do
    ( cd "$HOME/vv/cudarun" &&
      LD_LIBRARY_PATH=. /mnt/c/prog/claude/voicevox_emu_cpp/"$OUT"/vvmload \
          ./libvoicevox_onnxruntime.so.1.17.3 ./0.vvm "$which" 2>/dev/null |
          grep -E "^ok    (predict|decode|session)" )
done

echo
echo "== emulated"
for which in predict_duration predict_intonation decode; do
    ./vvcudaemu --sysroot "$PWD/sysroot" "$OUT/vvmload" \
        /opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 /opt/vvcuda/0.vvm "$which" 2>&1 |
        grep -vE "^x86emu: open" | grep -E "^ok    (predict|decode|session)|FAIL"
done
