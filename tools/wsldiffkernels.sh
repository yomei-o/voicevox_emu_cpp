#!/bin/sh
# The 377 kernels of a full utterance, native against emulated.
#
# The emulated run is deterministically wrong and AddressSanitizer is happy, so
# this is arithmetic rather than memory - and the first kernel whose output
# range differs says where.  The same method named the WebAssembly bug in one
# pass.
set -e
cd "$(dirname "$0")/.."
mkdir -p build

echo "== native"
( cd "$HOME/vv/cudarun" &&
  VVSTUB_STATS=1 LD_LIBRARY_PATH=. ./cudavvm ./libvoicevox_onnxruntime.so.1.17.3 \
      ./open_jtalk_dic_utf_8-1.11 ./0.vvm 3 "ずんだもんなのだ" /tmp/n.wav 2>&1 >/dev/null |
      grep '^\[k\]' ) > build/k_native.txt || true
wc -l < build/k_native.txt

echo "== emulated"
OUT=sysroot/opt/vvcuda
printf '%s' "ずんだもんなのだ" > "$OUT/text.txt"
VVSTUB_STATS=1 ./vvcudaemu --sysroot "$PWD/sysroot" "$OUT/cudavvm" \
    /opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 \
    /opt/vvcuda/open_jtalk_dic_utf_8-1.11 \
    /opt/vvcuda/0.vvm 3 @/opt/vvcuda/text.txt /opt/vvcuda/out.wav 2>&1 >/dev/null |
    grep '^\[k\]' > build/k_emu.txt || true
wc -l < build/k_emu.txt

echo
echo "== the first ten differences"
diff build/k_native.txt build/k_emu.txt | head -22
