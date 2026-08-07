#!/bin/sh
# The arguments of the launch that crashes, as bytes.
#
# The emulated run stops at its sixth kernel.  Whether the pointers in that
# launch are arena addresses (0x3000...), guest addresses (0x5555.../0x7fff...)
# or host ones says which of them was not converted, and the bytes say it
# outright.
cd "$(dirname "$0")/.."
OUT=sysroot/opt/vvcuda
printf '%s' "ずんだもんなのだ" > "$OUT/text.txt"
VVSTUB_KARGS=1 VVSTUB_STATS=1 ./vvcudaemu --sysroot "$PWD/sysroot" "$OUT/cudavvm" \
    /opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 \
    /opt/vvcuda/open_jtalk_dic_utf_8-1.11 \
    /opt/vvcuda/0.vvm 3 @/opt/vvcuda/text.txt /opt/vvcuda/out.wav 2>&1 |
    grep -vE "^x86emu: open" | tail -40
