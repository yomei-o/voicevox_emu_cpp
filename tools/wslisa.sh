#!/bin/sh
# The instruction-level regression, run against a Linux build of the emulator.
#
# The point is to be able to check the emulator while a Windows run is holding
# x86emu.exe open - and a second compiler on the same source is worth having
# anyway, for the same reason the wasm check is.
set -e
cd "$(dirname "$0")/.."
mkdir -p build
"${CXX:-$HOME/gpp/bin/g++}" -std=c++17 -O2 -Ix86_emu_cpp/src -o build/x86emu_linux \
    x86_emu_cpp/src/*.cpp
EMU=build/x86emu_linux sh tools/check_isa.sh
