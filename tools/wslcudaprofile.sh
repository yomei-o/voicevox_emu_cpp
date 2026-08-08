#!/bin/sh
# Where the CUDA run's instructions go, now that the cheap wins are taken.
#
# The last profile was taken before the upstream sync and the fetch cursor, and
# said 83 % ONNX Runtime / 12 % libvoicevox_core / 3.5 % libc.  That is the
# number that closed off hooking memcpy natively, so it is worth knowing whether
# it still holds before the next idea is chosen on the strength of it.
#
# Sampling every 100000 instructions: 7.8 G instructions is 78000 samples, which
# is plenty to rank modules and costs nothing measurable.
set -e
cd "$(dirname "$0")/.."
X86EMU_PROFILE=${X86EMU_PROFILE:-100000} \
    sh tools/wslrun_cuda.sh "${1:-ずんだもんなのだ}" 2>&1 |
    sed -n '/profile/,$p' | head -40
