#!/bin/sh
# Does the ONNX Runtime load and initialise inside the emulator?  No model.
set -e
cd "$(dirname "$0")"
EMU=${EMU:-../x86_emu_cpp/x86emu.exe}
mkdir -p sysroot/opt/vv
cp guest/probe guest/libvoicevox_onnxruntime.so.* sysroot/opt/vv/
export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1
exec "$EMU" --sysroot sysroot sysroot/opt/vv/probe \
    /opt/vv/libvoicevox_onnxruntime.so.1.17.3
