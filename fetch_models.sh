#!/bin/sh
# Fetches everything the guest needs into guest/, which is staged into the
# sysroot as /opt/vv.  None of it is redistributable, so none of it is
# committed; this script is the record of where each piece came from.
set -e
cd "$(dirname "$0")"

VV_CORE=0.16.4          # VOICEVOX/voicevox_core
VV_ORT=1.17.3           # VOICEVOX/onnxruntime-builder
VV_VVM=0.16.4           # VOICEVOX/voicevox_vvm
VVM=${1:-0.vvm}         # 0.vvm holds ずんだもん (styles 1, 3, 5, 7)

mkdir -p dl guest

get() { [ -f "dl/$2" ] || { echo "fetch $2"; curl -sL -o "dl/$2" "$1"; }; }

get "https://github.com/VOICEVOX/onnxruntime-builder/releases/download/voicevox_onnxruntime-$VV_ORT/voicevox_onnxruntime-linux-x64-$VV_ORT.tgz" ort.tgz
get "https://github.com/VOICEVOX/voicevox_core/releases/download/$VV_CORE/voicevox_core-linux-x64-$VV_CORE.zip" core.zip
get "https://github.com/VOICEVOX/voicevox_vvm/releases/download/$VV_VVM/$VVM" "$VVM"
get "https://sourceforge.net/projects/open-jtalk/files/Dictionary/open_jtalk_dic-1.11/open_jtalk_dic_utf_8-1.11.tar.gz/download" ojdic.tar.gz

# The .so is a symlink in the tarball; Windows cannot make one, so the
# versioned file is what everything refers to by name.
tar xzf dl/ort.tgz -C dl 2>/dev/null || true
cp "dl/voicevox_onnxruntime-linux-x64-$VV_ORT/lib/libvoicevox_onnxruntime.so.$VV_ORT" guest/

unzip -oq dl/core.zip -d dl
cp "dl/voicevox_core-linux-x64-$VV_CORE/lib/libvoicevox_core.so" guest/
cp "dl/voicevox_core-linux-x64-$VV_CORE/include/voicevox_core.h" src/

cp "dl/$VVM" guest/

tar xzf dl/ojdic.tar.gz -C dl
# 8 files including a 100 MB sys.dic; a short one means the download was cut off.
[ -f dl/open_jtalk_dic_utf_8-1.11/sys.dic ] || { echo "dictionary incomplete"; exit 1; }
rm -rf guest/open_jtalk_dic_utf_8-1.11
cp -r dl/open_jtalk_dic_utf_8-1.11 guest/

# The ONNX Runtime C API header, for probe.c.  MIT, from the upstream project
# the VOICEVOX build is a fork of.
[ -f src/onnxruntime_c_api.h ] || curl -sL -o src/onnxruntime_c_api.h \
    "https://raw.githubusercontent.com/microsoft/onnxruntime/v$VV_ORT/include/onnxruntime/core/session/onnxruntime_c_api.h"

echo
ls -la guest/
echo "models ready"
