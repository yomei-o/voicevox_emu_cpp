#!/bin/sh
# What each phase of a run costs, emulated.
#
# Module-level profiling says libvoicevox_core is 12 % of the instructions, but
# Open JTalk is statically linked into it, so that number covers the dictionary
# load, the text analysis and the decryption all at once.  Timing the phases
# separates them without needing symbols.
set -e
cd "$(dirname "$0")/.."
( cd "$HOME/vv/cudarun" &&
  gcc -O2 -I/mnt/c/prog/claude/voicevox_emu_cpp/src -o cudavvm \
      /mnt/c/prog/claude/voicevox_emu_cpp/src/cudavvm.c \
      -L. -l:libvoicevox_core.so -ldl )
cp "$HOME/vv/cudarun/cudavvm" guest/cudavvm

echo "== native, for scale"
( cd "$HOME/vv/cudarun" &&
  LD_LIBRARY_PATH=. ./cudavvm ./libvoicevox_onnxruntime.so.1.17.3 \
      ./open_jtalk_dic_utf_8-1.11 ./0.vvm 3 "ずんだもんなのだ" /tmp/n.wav 2>/dev/null |
      grep -E "^step|^      \.\.\.|tts took" )

echo
echo "== emulated"
sh tools/wslrun_cuda.sh "ずんだもんなのだ" 2>&1 |
    grep -vE "^x86emu: open|^kernel " | grep -E "^step|^      \.\.\.|tts took"
