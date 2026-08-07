#!/bin/sh
# Rebuild both halves and check both still produce the T4's audio.
#
# The two builds share source and share a name space, so this runs both rather
# than trusting that a change to one left the other alone.
set -e
cd "$(dirname "$0")/.."

echo "== native"
sh tools/wslcheck_native.sh 2>&1 | tail -5

echo
echo "== guest, under the emulator"
sh tools/wslstubs.sh > /dev/null 2>&1
sh tools/wslbuild.sh > /dev/null 2>&1
rm -f sysroot/opt/vvcuda/out.wav
sh tools/wslrun_cuda.sh "ずんだもんなのだ" 2>&1 |
    grep -vE "^x86emu: open|^kernel " | grep -E "tts took|WARNING|launches"
[ -f sysroot/opt/vvcuda/out.wav ] || { echo "the run did not finish"; exit 1; }
python3 tools/wavcmp.py sysroot/opt/vvcuda/out.wav web/sample/gpu_zundamon.wav | tail -3
