#!/bin/sh
# From a fresh clone to a runnable tree.  Downloads nothing: the runtime, the
# core, the voice model, the dictionary and a Debian sysroot are all in the
# repository (see licenses/README.md for what is redistributed under what).
#
#   sh setup.sh          # unpack, build the emulator, build the guests
#   sh run_probe.sh      # does the runtime come up?      (~10 s)
#   sh run_tts.sh        # the whole thing                (see resume.md)
#
# To change the voice model or refresh the runtime, use fetch_models.sh; to
# rebuild the sysroot from Debian's packages, make_sysroot.sh.  Neither is
# needed for an ordinary clone.
set -e
cd "$(dirname "$0")"

echo "=== 1/3  unpacking the payload"
sh unpack.sh

echo "=== 2/3  emulator"
# The vendored copy, not a sibling checkout: the fixes this needs live here
# until they are merged back, and a sibling built before them fails in the
# original, very confusing way.  See emu_path.sh.
if [ -x x86_emu_cpp/build/Release/x86emu.exe ] || [ -x x86_emu_cpp/x86emu.exe ] ||
   [ -x x86_emu_cpp/x86emu ]; then
    echo "    already built"
elif command -v "${CXX:-g++}" >/dev/null 2>&1; then
    (cd x86_emu_cpp && sh build.sh)
else
    # No g++, which on Windows is the ordinary case: build with MSVC through
    # CMake instead.  vcvars is deliberately not used - on the machine this was
    # written on it hangs forever when invoked from a shell - and CMake's
    # Visual Studio generator does not need it.
    CMAKE=${CMAKE:-cmake}
    command -v "$CMAKE" >/dev/null 2>&1 || CMAKE="C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
    [ -x "$CMAKE" ] || command -v "$CMAKE" >/dev/null 2>&1 || {
        echo "    no g++ and no cmake: install one, or set CXX= / CMAKE="
        exit 1
    }
    echo "    no g++; building with cmake and MSVC"
    (cd x86_emu_cpp && "$CMAKE" -B build -G "Visual Studio 17 2022" -A x64 > /dev/null &&
        "$CMAKE" --build build --config Release --parallel)
fi

echo "=== 3/3  guests"
sh build.sh

echo
echo "ready.  sh run_probe.sh"
