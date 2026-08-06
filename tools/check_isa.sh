#!/bin/sh
# The instruction-level regression check, in one command.
#
#   sh tools/check_isa.sh              # the native emulator
#   sh tools/check_isa.sh wasm         # the WebAssembly build too
#
# Runs src/isatest.c under the emulator and diffs the 240 group checksums
# against qemu_ref.txt, which is what a real CPU and qemu-x86_64 both produce.
# Run it after touching the emulator, and run the wasm side as well: a
# WebAssembly build is a different compiler on a different target, and
# undefined behaviour that happens to be right on x86 is not right there.  That
# is how the CVTPS2DQ bug was found.
#
# To regenerate the reference on a Linux x86-64 machine:
#
#   gcc -O2 -o isatest src/isatest.c
#   ./isatest > native.txt && qemu-x86_64 ./isatest > qemu_ref.txt
#   diff native.txt qemu_ref.txt        # must be empty, or the masks are wrong
set -e
cd "$(dirname "$0")/.."
what=${1:-native}

[ -f sysroot/opt/vv/isatest ] || {
    [ -f guest/isatest ] || { echo "build guest/isatest from src/isatest.c first"; exit 1; }
    mkdir -p sysroot/opt/vv
    cp guest/isatest sysroot/opt/vv/
}
export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1

fail=0
report() {  # report <label> <output file>
    if diff -q qemu_ref.txt "$2" > /dev/null 2>&1; then
        echo "ok    $1: 240/240 identical to native and qemu-x86_64"
    else
        echo "FAIL  $1:"
        diff qemu_ref.txt "$2" | head -20
        fail=1
    fi
}

if [ "$what" != "wasm" ]; then
    emu=${EMU:-}
    [ -n "$emu" ] || for c in x86_emu_cpp/build/Release/x86emu.exe x86_emu_cpp/x86emu.exe \
                              x86_emu_cpp/x86emu ../x86_emu_cpp/x86emu.exe; do
        [ -x "$c" ] && { emu=$c; break; }
    done
    [ -n "$emu" ] || { echo "no emulator built; set EMU="; exit 1; }
    "$emu" --sysroot sysroot sysroot/opt/vv/isatest > build/isatest_native.txt 2>&1 || true
    report "native ($emu)" build/isatest_native.txt
fi

if [ "$what" = "wasm" ] || [ "$what" = "all" ]; then
    NODE=${NODE:-node}
    command -v "$NODE" >/dev/null 2>&1 || { echo "no node; set NODE="; exit 1; }
    "$NODE" web/test_node.mjs isatest > build/isatest_wasm.txt 2>/dev/null || true
    report "wasm" build/isatest_wasm.txt
fi

exit $fail
