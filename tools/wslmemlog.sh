#!/bin/sh
# The guest-visible part of the last tools/wslmem.sh run.
grep -vE "^x86emu: open|^kernel |^	" "$HOME/vvmem.log" | head -30
