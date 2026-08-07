#!/bin/sh
# What the emulated CUDA run costs in memory, and whether it still works.
#
# The provider is 460 MB and 419 MB of it is device code nothing here executes.
# With pages reserved rather than allocated, and a page of zeros over an
# untouched page skipped, none of that should be resident.
#
# The output is deleted first: a run that stops early otherwise leaves the
# previous one in place, and comparing that proves nothing.
cd "$(dirname "$0")/.."
rm -f sysroot/opt/vvcuda/out.wav
/usr/bin/time -v sh tools/wslrun_cuda.sh "ずんだもんなのだ" > "$HOME/vvmem.log" 2>&1
grep -vE "^x86emu: open|^kernel " "$HOME/vvmem.log" | tail -30
echo "== memory"
grep -E "Maximum resident|Elapsed" "$HOME/vvmem.log"
[ -f sysroot/opt/vvcuda/out.wav ] || { echo "no out.wav - the run did not finish"; exit 1; }
python3 tools/wavcmp.py sysroot/opt/vvcuda/out.wav web/sample/gpu_zundamon.wav | tail -3
