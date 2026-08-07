#!/bin/sh
# The same launch's arguments, native against emulated, byte for byte.
#
# The emulated run reads its fifth argument as a pointer where the native run
# reads the element count, and both are the same ORT binary building the same
# argument array.  Only one of those readings can be right, and the bytes of
# both, side by side, say which.  The full mangled name comes along because the
# template flags decide the signature.
set -e
cd "$(dirname "$0")/.."
mkdir -p build

# Six kernels is as far as the emulated run gets, so that is the window.
window() { grep -E '^ *(_|[A-Za-z].*, [0-9]+ args)|^ *\[[0-9]+\]' | head -60; }

echo "== rebuilding the native shim"
CXX=$HOME/gpp/bin/g++ CC=gcc MODE=native \
    sh tools/make_cuda_stubs.sh \
    "$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib" > "$HOME/vvnative.log" 2>&1 ||
    { tail -20 "$HOME/vvnative.log"; exit 1; }
cp guest/cudastub/*.so.* "$HOME/vv/cudarun/"

echo "== rebuilding the emulator"
sh tools/wslbuild.sh > "$HOME/vvbuild.log" 2>&1 || { tail -20 "$HOME/vvbuild.log"; exit 1; }

echo "== native"
( cd "$HOME/vv/cudarun" &&
  VVSTUB_KARGS=1 LD_LIBRARY_PATH=. ./cudavvm ./libvoicevox_onnxruntime.so.1.17.3 \
      ./open_jtalk_dic_utf_8-1.11 ./0.vvm 3 "ずんだもんなのだ" /tmp/n.wav 2>/dev/null ) |
  window > build/a_native.txt || true

echo "== emulated"
OUT=sysroot/opt/vvcuda
printf '%s' "ずんだもんなのだ" > "$OUT/text.txt"
VVSTUB_KARGS=1 ./vvcudaemu --sysroot "$PWD/sysroot" "$OUT/cudavvm" \
    /opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 \
    /opt/vvcuda/open_jtalk_dic_utf_8-1.11 \
    /opt/vvcuda/0.vvm 3 @/opt/vvcuda/text.txt /opt/vvcuda/out.wav 2>/dev/null |
    window > build/a_emu.txt || true

echo
echo "== the launches, in order"
paste -d'\n' /dev/null /dev/null > /dev/null 2>&1 || true
grep 'args$' build/a_native.txt | head -8 | sed 's/^/native   /'
echo
grep 'args$' build/a_emu.txt | head -8 | sed 's/^/emulated /'
echo
echo "== the sixth launch, both sides"
awk '/BinaryElementWiseSimple/ {f=1} f&&c<6 {print; if(/^ *\[/) c++}' build/a_native.txt |
    sed 's/^/N /'
echo
awk '/BinaryElementWiseSimple/ {f=1} f&&c<6 {print; if(/^ *\[/) c++}' build/a_emu.txt |
    sed 's/^/E /'
