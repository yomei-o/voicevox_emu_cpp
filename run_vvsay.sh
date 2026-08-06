#!/bin/sh
# Text in, a WAV out, through the drop-in API.
#
#   sh build_api.sh        # once: guest/vvagent and voicevox_core.dll
#   sh run_vvsay.sh                       # says text.txt
#   sh run_vvsay.sh text_short.txt        # says something else
#
# vvsay is an ordinary program written against the official voicevox_core.h; it
# does not know there is an emulator underneath.  What makes that work is the
# environment set below.
#
# Expect this to take about an hour: eight minutes to decrypt the model and
# initialise the sessions, then roughly eighty times the length of the audio.
set -e
cd "$(dirname "$0")"

TEXT=${1:-text.txt}
OUT=${2:-out.wav}
STYLE=${3:-3}
VVM=${VVM:-0.vvm}

[ -x ./vvsay.exe ] || [ -x ./vvsay ] || {
    echo "build vvsay first (see build_api.sh, then compile src/vvsay.c against"
    echo "voicevox_core.lib)"
    exit 1
}
[ -f sysroot/opt/vv/vvagent ] || { echo "run: sh build_api.sh"; exit 1; }
sh unpack.sh
mkdir -p sysroot/opt/vv
cp guest/libvoicevox_core.so guest/libvoicevox_onnxruntime.so.* "guest/$VVM" sysroot/opt/vv/
[ -d sysroot/opt/vv/open_jtalk_dic_utf_8-1.11 ] ||
    cp -r guest/open_jtalk_dic_utf_8-1.11 sysroot/opt/vv/

# MSYS rewrites any argument that looks like a Unix path into a Windows one, and
# the guest's paths look exactly like that: /opt/vv/0.vvm would arrive as
# C:/Program Files/Git/opt/vv/0.vvm and fail to open.
export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1

# Where the host library finds the emulator and the guest's filesystem.
#
# These are read by a *Windows* program, so they have to be Windows paths.  MSYS
# normally rewrites them on the way out, but MSYS_NO_PATHCONV above - which the
# guest's own arguments need - turns that off for everything, so convert them
# here.  Without this the library is handed /c/prog/... and reports, quite
# correctly, that there is no emulator there.
win() { cygpath -m "$1" 2>/dev/null || echo "$1"; }

for cand in x86_emu_cpp/build/Release/x86emu.exe x86_emu_cpp/x86emu.exe x86_emu_cpp/x86emu; do
    [ -x "$cand" ] && { found=$cand; break; }
done
[ -n "$VOICEVOX_EMU_EMULATOR" ] || VOICEVOX_EMU_EMULATOR=$(win "$PWD/$found")
[ -n "$VOICEVOX_EMU_ROOT" ] || VOICEVOX_EMU_ROOT=$(win "$PWD")
export VOICEVOX_EMU_EMULATOR VOICEVOX_EMU_ROOT
echo "emulator: $VOICEVOX_EMU_EMULATOR"

VVSAY=./vvsay.exe
[ -x "$VVSAY" ] || VVSAY=./vvsay
exec "$VVSAY" /opt/vv/libvoicevox_onnxruntime.so.1.17.3 \
    /opt/vv/open_jtalk_dic_utf_8-1.11 "/opt/vv/$VVM" "$TEXT" "$STYLE" "$OUT"
