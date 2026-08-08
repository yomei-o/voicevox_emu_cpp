#!/bin/sh
# Two emulator builds, on the CUDA path, interleaved.
#
#   sh tools/wslcudaab.sh [rounds]
#
# Interleaved and not one-after-the-other, because this machine's wall clock
# moves 10-15 % on its own: three runs of A then three of B measures whatever
# else the machine was doing as much as it measures the change.  A, B, A, B does
# not.
#
# It reports the two numbers that matter separately.  Session build is the whole
# cost of a CUDA run - decrypting the model and constructing the graphs, all of
# it interpreted.  Synthesis is seconds, because the arithmetic is over the shim
# on the host, so a change to the interpreter cannot move it much and a change
# that appears to has probably measured noise.
set -e
cd "$(dirname "$0")/.."
ROUNDS=${1:-3}
[ -x ./vvcudaemu.before ] || { echo "no ./vvcudaemu.before to compare against"; exit 1; }
[ -x ./vvcudaemu ] || { echo "no ./vvcudaemu - run tools/wslbuild.sh"; exit 1; }

one() {  # one <label> <binary>
    out=$(VVEMU=$2 sh tools/wslrun_cuda.sh "ずんだもんなのだ" 2>&1) || {
        echo "  $1: FAILED"
        printf '%s\n' "$out" | tail -5 | sed 's/^/      /'
        return 1
    }
    build=$(printf '%s\n' "$out" | grep -A1 load_voice_model | sed -n 's/.*\.\.\. \([0-9.]*\) s.*/\1/p' | head -1)
    tts=$(printf '%s\n' "$out" | sed -n 's/.*tts took \([0-9.]*\) s.*/\1/p' | head -1)
    printf '  %-8s session %8s s   tts %7s s\n' "$1" "${build:-?}" "${tts:-?}"
    # The audio, so that a faster run that is also a different run is caught
    # here rather than believed.
    cp sysroot/opt/vvcuda/out.wav "build/ab_$1.wav" 2>/dev/null || true
}

mkdir -p build
r=1
while [ "$r" -le "$ROUNDS" ]; do
    echo "round $r"
    # presync is optional: it is the emulator as it was before the upstream sync,
    # and only exists when tools/wslbuildpresync.sh has been run.
    [ -x ./vvcudaemu.presync ] && one presync ./vvcudaemu.presync
    one before ./vvcudaemu.before
    one after  ./vvcudaemu
    r=$((r + 1))
done
