#!/bin/sh
# Builds the two halves of the drop-in API.
#
#   guest/vvagent          the guest agent: Linux x86-64, links the official
#                          libvoicevox_core.so, runs inside the emulator
#   voicevox_core.dll      the host library: exports VOICEVOX CORE's C API and
#   (or libvoicevox_core_emu.so)   answers each call by asking the agent
#
# A program written against the official `voicevox_core.h` links the host
# library instead and runs unchanged.
#
#   sh build_api.sh              # both halves
#   sh build_api.sh guest        # just the guest
#   sh build_api.sh host         # just the host
#
# The guest needs a cross compiler for x86-64 Linux: clang with lld as in
# build.sh, or - if this machine has WSL - the gcc inside it, which is simpler
# and is what WSL=1 selects.
set -e
cd "$(dirname "$0")"
what=${1:-both}

build_guest() {
    echo "== guest agent"
    if [ -n "$WSL" ] || { [ -z "$CLANG" ] && [ ! -x "/c/Program Files/LLVM/bin/clang.exe" ] && command -v wsl.exe >/dev/null 2>&1; }; then
        # WSL's own gcc builds a native x86-64 Linux binary, no sysroot needed.
        DISTRO=${WSL_DISTRO:-Ubuntu-22.04}
        HERE=$(pwd)
        wsl.exe -d "$DISTRO" -- bash -c "cd /mnt/\$(echo '$HERE' | sed 's|^/\\([a-zA-Z]\\)/|\\1/|' | tr 'A-Z' 'a-z') && gcc -O2 -Wall -Wextra -Isrc -o guest/vvagent src/vvagent.c -Lguest -lvoicevox_core -Wl,-rpath,/opt/vv"
    else
        CLANG=${CLANG:-/c/Program Files/LLVM/bin/clang.exe}
        [ -x "$CLANG" ] || { echo "set CLANG, or set WSL=1 to build the guest in WSL"; exit 1; }
        ROOT=$(cygpath -w "$PWD/sysroot" 2>/dev/null || echo "$PWD/sysroot")
        export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1
        "$CLANG" --target=x86_64-unknown-linux-gnu --sysroot="$ROOT" -fuse-ld=lld \
            -B "$ROOT/usr/lib/gcc/x86_64-linux-gnu/12" -O2 -Wall -Isrc \
            -o guest/vvagent src/vvagent.c -Lguest -lvoicevox_core -Wl,-rpath,/opt/vv
    fi
    mkdir -p sysroot/opt/vv
    cp guest/vvagent sysroot/opt/vv/vvagent
    echo "   guest/vvagent -> sysroot/opt/vv/vvagent"
}

build_host() {
    echo "== host library"
    case "$(uname -s)" in
        MINGW* | MSYS* | CYGWIN*)
            VC=${VC:-C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.31.31103}
            SDK=${SDK:-C:/Program Files (x86)/Windows Kits/10}
            SDKVER=${SDKVER:-10.0.19041.0}
            CL="$VC/bin/Hostx64/x64/cl.exe"
            [ -x "$CL" ] || { echo "no cl.exe at $CL - set VC="; exit 1; }
            export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1 VSLANG=1033
            mkdir -p build/api
            # vcvars is not used: the include and library directories go in as
            # flags, which is the only way that works reliably here.
            "$CL" /nologo /LD /O2 /W3 /utf-8 /D_CRT_SECURE_NO_WARNINGS \
                "/I$VC/include" "/I$SDK/Include/$SDKVER/ucrt" \
                "/I$SDK/Include/$SDKVER/um" "/I$SDK/Include/$SDKVER/shared" \
                /Isrc src/vvhost.c \
                /Fo:build/api/ /Fe:voicevox_core.dll \
                /link "/LIBPATH:$VC/lib/x64" "/LIBPATH:$SDK/Lib/$SDKVER/ucrt/x64" \
                "/LIBPATH:$SDK/Lib/$SDKVER/um/x64"
            echo "   voicevox_core.dll"
            ;;
        *)
            CC=${CC:-cc}
            $CC -O2 -Wall -Wextra -fPIC -shared -Isrc -o libvoicevox_core_emu.so src/vvhost.c \
                -lpthread
            echo "   libvoicevox_core_emu.so"
            ;;
    esac
}

case "$what" in
    guest) build_guest ;;
    host) build_host ;;
    *)
        build_guest
        build_host
        ;;
esac
echo "done"
