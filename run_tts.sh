#!/bin/sh
# Text in, sysroot/opt/vv/out.wav out - through the official CORE, emulated.
set -e
cd "$(dirname "$0")"
. "$(dirname "$0")/emu_path.sh"
VVM=${VVM:-0.vvm}
sh "$(dirname "$0")/unpack.sh"
mkdir -p sysroot/opt/vv
cp guest/tts guest/libvoicevox_core.so guest/libvoicevox_onnxruntime.so.* "guest/$VVM" sysroot/opt/vv/
[ -d sysroot/opt/vv/open_jtalk_dic_utf_8-1.11 ] || cp -r guest/open_jtalk_dic_utf_8-1.11 sysroot/opt/vv/
export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1
exec "$EMU" --sysroot sysroot sysroot/opt/vv/tts \
    /opt/vv/libvoicevox_onnxruntime.so.1.17.3 \
    /opt/vv/open_jtalk_dic_utf_8-1.11 \
    "/opt/vv/$VVM"
