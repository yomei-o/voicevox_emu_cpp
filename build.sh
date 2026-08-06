#!/bin/sh
# Cross-compiles the guests to x86-64 Linux with clang and lld, against the
# sysroot make_sysroot.sh built.  Runs on any host clang runs on - there is no
# Linux and no WSL in this loop.
#
# Where WSL is available its gcc builds the same programs natively and needs no
# sysroot at all, which is simpler; WSL=1 takes that path.  The guest agent for
# the drop-in API is built by build_api.sh, which shares this choice.
set -e
cd "$(dirname "$0")"

# The programs, and whether each one links CORE.
#   probe     does the runtime load and initialise at all
#   tts       the whole pipeline: text in, out.wav out
#   analyze   text in, accent phrases out - no inference, so it is fast
#   isatest   the instruction differential test (links nothing)
#   memtest   large-buffer allocator test (links nothing)
WITH_CORE="tts analyze"
STANDALONE="probe isatest memtest"

mkdir -p guest

if [ -n "$WSL" ] || { [ -z "$CLANG" ] && [ ! -x "/c/Program Files/LLVM/bin/clang.exe" ] &&
                      command -v wsl.exe >/dev/null 2>&1; }; then
    DISTRO=${WSL_DISTRO:-Ubuntu-22.04}
    HERE=$(pwd)
    # /c/prog/x -> /mnt/c/prog/x, which is how WSL sees this directory.
    WHERE=/mnt/$(echo "$HERE" | sed 's|^/\([a-zA-Z]\)/|\1/|' | tr 'A-Z' 'a-z')
    for p in $STANDALONE; do
        echo "== $p (wsl gcc)"
        wsl.exe -d "$DISTRO" -- bash -c \
            "cd $WHERE && gcc -O2 -Wall -Isrc -o guest/$p src/$p.c \$( [ $p = probe ] && echo -ldl )"
    done
    for p in $WITH_CORE; do
        echo "== $p (wsl gcc)"
        wsl.exe -d "$DISTRO" -- bash -c \
            "cd $WHERE && gcc -O2 -Wall -Isrc -o guest/$p src/$p.c -Lguest -lvoicevox_core -Wl,-rpath,/opt/vv"
    done
    echo "done"
    exit 0
fi

CLANG=${CLANG:-/c/Program Files/LLVM/bin/clang.exe}
[ -x "$CLANG" ] || { echo "set CLANG to a clang with an lld beside it, or WSL=1"; exit 1; }
[ -d sysroot ] || { echo "run make_sysroot.sh first"; exit 1; }

# clang wants a Windows-shaped --sysroot on a Windows host, and MSYS must be
# stopped from rewriting the guest paths in -Wl,-rpath into C:/... .
ROOT=$(cygpath -w "$PWD/sysroot" 2>/dev/null || echo "$PWD/sysroot")
export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1

cc() {
    "$CLANG" --target=x86_64-unknown-linux-gnu --sysroot="$ROOT" -fuse-ld=lld \
        -B "$ROOT/usr/lib/gcc/x86_64-linux-gnu/12" -O2 -Wall -Isrc "$@"
}

for p in $STANDALONE; do
    echo "== $p"
    if [ "$p" = probe ]; then
        cc -o "guest/$p" "src/$p.c" -ldl
    else
        cc -o "guest/$p" "src/$p.c"
    fi
done
for p in $WITH_CORE; do
    echo "== $p"
    cc -o "guest/$p" "src/$p.c" -Lguest -lvoicevox_core -Wl,-rpath,/opt/vv
done

echo "done"
