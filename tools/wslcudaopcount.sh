#!/bin/sh
# Which instructions a CUDA run actually executes.
#
# The sampling profiler has said "83 % ONNX Runtime" through three rounds of
# changes.  That is true and names nothing to fix.  This counts opcodes, so the
# next optimisation can be aimed at the handlers that run rather than the ones
# that look expensive.
set -e
cd "$(dirname "$0")/.."
X86EMU_OPCOUNT=1 sh tools/wslrun_cuda.sh "${1:-ずんだもんなのだ}" 2>&1 |
    sed -n '/opcount/p' | head -40
