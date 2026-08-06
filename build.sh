#!/bin/sh
# Cross-compiles the two guests to x86-64 Linux with clang and lld, against the
# sysroot make_sysroot.sh built.  Runs on any host clang runs on - there is no
# Linux and no WSL in this loop.
set -e
cd "$(dirname "$0")"

CLANG=${CLANG:-/c/Program Files/LLVM/bin/clang.exe}
[ -x "$CLANG" ] || { echo "set CLANG to a clang with an lld beside it"; exit 1; }
[ -d sysroot ] || { echo "run make_sysroot.sh first"; exit 1; }

# clang wants a Windows-shaped --sysroot on a Windows host, and MSYS must be
# stopped from rewriting the guest paths in -Wl,-rpath into C:/... .
ROOT=$(cygpath -w "$PWD/sysroot" 2>/dev/null || echo "$PWD/sysroot")
export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1

cc() {
    "$CLANG" --target=x86_64-unknown-linux-gnu --sysroot="$ROOT" -fuse-ld=lld \
        -B "$ROOT/usr/lib/gcc/x86_64-linux-gnu/12" -O2 -Wall "$@"
}

mkdir -p guest
echo "== probe"
cc -o guest/probe src/probe.c

echo "== tts"
cc -o guest/tts src/tts.c -Lguest -lvoicevox_core -Wl,-rpath,/opt/vv

echo "done"
