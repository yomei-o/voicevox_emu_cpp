#!/bin/sh
# Where the emulated run's instructions go, by mapping.
#
# Synthesis through the CUDA shim takes seconds now, and building the sessions
# takes six and a half minutes.  That inversion is what this is for: the setup
# is the whole cost of a run, and "which library" is usually most of the answer.
#
# One sample every 100k instructions.  A run retires tens of billions, so that
# is a few hundred thousand samples - plenty, and small enough to keep.
cd "$(dirname "$0")/.."
sh tools/wslbuild.sh > /dev/null 2>&1 || { sh tools/wslbuild.sh; exit 1; }
rm -f sysroot/opt/vvcuda/out.wav
X86EMU_PROFILE=${X86EMU_PROFILE:-100000} sh tools/wslrun_cuda.sh "あ" > "$HOME/vvprof.log" 2>&1
grep -E "^\[profile\]" "$HOME/vvprof.log" | head -20
echo "== and the run itself"
grep -vE "^x86emu: open|^kernel |^\[profile\]" "$HOME/vvprof.log" | grep -E "tts took|produced|WARNING"
