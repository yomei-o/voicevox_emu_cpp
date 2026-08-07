#!/bin/sh
# The profile from the last tools/wslprofile.sh run, however far it has got.
grep -E "^\[profile\]" "$HOME/vvprof.log" | head -20
echo "== the run"
grep -vE "^x86emu: open|^kernel |^\[profile\]" "$HOME/vvprof.log" |
    grep -E "tts took|produced|WARNING|load_voice_model|ok |step " | tail -8
