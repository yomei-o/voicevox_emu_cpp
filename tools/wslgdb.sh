#!/bin/sh
# Where the full run crashes, with a backtrace.
#
# Guessing has had two turns at this already.  It takes five minutes to reach
# the fault, which is a long time to spend on a hypothesis and no time at all to
# spend on an answer.
set -e
cd "$(dirname "$0")/.."
command -v gdb > /dev/null 2>&1 || { echo "no gdb - apt install gdb"; exit 1; }

OUT=sysroot/opt/vvcuda
printf '%s' "ずんだもんなのだ" > "$OUT/text.txt"
rm -f "$OUT/out.wav"

gdb -q --batch -ex run -ex bt -ex 'info locals' --args \
    ./vvcudaemu --sysroot "$PWD/sysroot" "$OUT/cudavvm" \
    /opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 \
    /opt/vvcuda/open_jtalk_dic_utf_8-1.11 \
    /opt/vvcuda/0.vvm 3 @/opt/vvcuda/text.txt /opt/vvcuda/out.wav 2>&1 |
    grep -vE "^x86emu: open|^kernel " | tail -40
