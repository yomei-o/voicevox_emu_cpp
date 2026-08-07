#!/bin/sh
# The emulated CUDA pipeline, end to end, against the audio the T4 produced.
#
# regress_tts.sh is the same check for the CPU build; this is the one for the
# shim.  It deletes the previous out.wav first for the same reason: a run that
# is cut short leaves the old file in place, and comparing that against the
# reference reports zero difference and means nothing.
set -e
cd "$(dirname "$0")/.."
rm -f sysroot/opt/vvcuda/out.wav
start=$(date +%s)
sh tools/wslrun_cuda.sh "ずんだもんなのだ" 2>&1 | grep -vE "^x86emu: open|^kernel " | tail -12
echo "elapsed $(( $(date +%s) - start )) s"
[ -f sysroot/opt/vvcuda/out.wav ] ||
    { echo "REGRESS: no out.wav - the run did not finish"; exit 1; }
python3 tools/wavcmp.py sysroot/opt/vvcuda/out.wav web/sample/gpu_zundamon.wav
