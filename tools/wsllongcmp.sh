#!/bin/sh
# The long utterance both ways, and compared.
#
# There is no recorded reference for this text, so the check is the native shim
# against the emulated one.  That is worth something: the native shim matches a
# Tesla T4 on both texts there *are* references for, so agreeing with it is
# agreeing with the T4 by transitivity - and the two runs share the arithmetic
# but not the path to it.
set -e
cd "$(dirname "$0")/.."
TEXT="ずんだもんなのだ。きょうはとてもいいてんきなので、こうえんまでさんぽにいってきました。とちゅうでねこにあったのだ。"

echo "== native"
( cd "$HOME/vv/cudarun" &&
  LD_LIBRARY_PATH=. ./cudavvm ./libvoicevox_onnxruntime.so.1.17.3 \
      ./open_jtalk_dic_utf_8-1.11 ./0.vvm 3 "$TEXT" long_native.wav 2>/dev/null |
      grep -E "tts took|produced" )

echo "== emulated"
rm -f sysroot/opt/vvcuda/out.wav
sh tools/wslrun_cuda.sh "$TEXT" 2>&1 |
    grep -vE "^x86emu: open|^kernel " | grep -E "tts took|produced|WARNING|launches"

echo
python3 tools/wavcmp.py sysroot/opt/vvcuda/out.wav "$HOME/vv/cudarun/long_native.wav"
