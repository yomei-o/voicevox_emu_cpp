#!/bin/sh
# Peak memory for the emulated CUDA run, before and after the paging changes.
#
# "Before" is the committed version of the two files that changed, built into a
# second binary; nothing is stashed and the working tree is left alone.  The
# comparison is the point: a claim about memory that is not measured both ways
# is not a measurement.
set -e
cd "$(dirname "$0")/.."
REPO=$PWD
BASE=${1:-HEAD}

run() {   # $1 = binary, $2 = label
    rm -f sysroot/opt/vvcuda/out.wav
    cp "$1" ./vvcudaemu
    /usr/bin/time -v sh tools/wslrun_cuda.sh "ずんだもんなのだ" > "$HOME/vvab.log" 2>&1 || true
    peak=$(sed -n 's/.*Maximum resident set size (kbytes): //p' "$HOME/vvab.log")
    wall=$(sed -n 's/.*Elapsed (wall clock) time.*: //p' "$HOME/vvab.log")
    ok=no
    [ -f sysroot/opt/vvcuda/out.wav ] && ok=yes
    printf '%-28s peak %7s kB   wall %-8s finished %s\n' "$2" "$peak" "$wall" "$ok"
}

mkdir -p build/ab
for f in memory.h memory.cpp syscalls.cpp; do
    git show "$BASE:x86_emu_cpp/src/$f" > "build/ab/$f"
done

echo "== building the committed version"
mkdir -p build/ab/src
cp x86_emu_cpp/src/*.h x86_emu_cpp/src/*.cpp x86_emu_cpp/src/*.inc build/ab/src/
cp build/ab/memory.h build/ab/memory.cpp build/ab/syscalls.cpp build/ab/src/
rm -f build/ab/src/main.cpp
"${CXX:-$HOME/gpp/bin/g++}" -std=c++17 -O2 -Ibuild/ab/src -Isrc -Ithird_party/eigen_flat \
    -o build/ab/vvcudaemu_before src/vvcudaemu.cpp build/ab/src/*.cpp \
    build/cudaemu/cudakernels.o build/cudaemu/cudnn_real.o \
    build/cudaemu/cublas_real.o build/cudaemu/cudahost.o

echo "== building the working tree"
CXX=${CXX:-$HOME/gpp/bin/g++} CC=gcc sh build_cudaemu.sh > "$HOME/vvab_build.log" 2>&1
cp vvcudaemu build/ab/vvcudaemu_after

echo
run build/ab/vvcudaemu_before "before (committed)"
run build/ab/vvcudaemu_after  "after (working tree)"
cp build/ab/vvcudaemu_after ./vvcudaemu
