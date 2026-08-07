#!/bin/sh
# The full run under AddressSanitizer.
#
# The crash is intermittent and the audio is wrong when it does finish, which is
# what an out-of-bounds access looks like from the outside: sometimes it lands
# in mapped memory and sometimes it does not.  Reading the code for it has had
# two turns and got one of them; this names the access.
set -e
cd "$(dirname "$0")/.."
CXX=${CXX:-$HOME/gpp/bin/g++}
CC=${CC:-gcc}
OPT="-O1 -g -fsanitize=address -fno-omit-frame-pointer"
INC="-Ix86_emu_cpp/src -Isrc -Ithird_party/eigen_flat"
EMU_SRC=$(ls x86_emu_cpp/src/*.cpp | grep -v '/main\.cpp$')

mkdir -p build/asan
echo "== building with AddressSanitizer"
$CC $OPT $INC -c src/cudakernels.c -o build/asan/cudakernels.o
$CXX -std=c++17 $OPT $INC -c src/cudnn_real.cpp -o build/asan/cudnn_real.o
$CXX -std=c++17 $OPT $INC -c src/cublas_real.cpp -o build/asan/cublas_real.o
$CXX -std=c++17 $OPT $INC -c src/cudahost.cpp -o build/asan/cudahost.o
$CXX -std=c++17 $OPT $INC -o build/asan/vvcudaemu src/vvcudaemu.cpp $EMU_SRC \
    build/asan/cudakernels.o build/asan/cudnn_real.o build/asan/cublas_real.o \
    build/asan/cudahost.o

OUT=sysroot/opt/vvcuda
printf '%s' "${1:-ずんだもんなのだ}" > "$OUT/text.txt"
rm -f "$OUT/out.wav"

echo "== running (slower than usual; that is what the checking costs)"
ASAN_OPTIONS=detect_leaks=0:abort_on_error=0 build/asan/vvcudaemu \
    --sysroot "$PWD/sysroot" "$OUT/cudavvm" \
    /opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 \
    /opt/vvcuda/open_jtalk_dic_utf_8-1.11 \
    /opt/vvcuda/0.vvm 3 @/opt/vvcuda/text.txt /opt/vvcuda/out.wav 2>&1 |
    grep -vE "^x86emu: open|^kernel " | tail -50
