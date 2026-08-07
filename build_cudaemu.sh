#!/bin/sh
# Builds vvcudaemu: the emulator with the CUDA shim wired in behind it.
#
# Two halves that never meet in the same address space anywhere else:
#
#   the emulator          x86_emu_cpp/src/*.cpp, minus its own main
#   the shim              src/cudahost.cpp (the boundary) plus the same
#                         arithmetic the native shim uses, unchanged
#
# The guest half is separate - `MODE=guest sh tools/make_cuda_stubs.sh` builds
# the stand-ins that forward across it.
#
# Unlike the guest stand-ins, this is host code: it may use whatever the host
# has, and the convolutions are 90 % of the time, so it does.
set -e
cd "$(dirname "$0")"
CXX=${CXX:-g++}
CC=${CC:-gcc}
OUT=${OUT:-vvcudaemu}
OPT=${OPT:--O2}
if [ "$OPT" = "-O2" ] && grep -q ' avx2 ' /proc/cpuinfo 2>/dev/null; then
    OPT="-O3 -mavx2 -mfma"
fi
echo "flags $OPT"

EMU_SRC=$(ls x86_emu_cpp/src/*.cpp | grep -v '/main\.cpp$')
INC="-Ix86_emu_cpp/src -Isrc -Ithird_party/eigen_flat"

mkdir -p build/cudaemu
echo "== the arithmetic (C)"
$CC $OPT $INC -c src/cudakernels.c -o build/cudaemu/cudakernels.o
echo "== the arithmetic (C++) and the boundary"
$CXX -std=c++17 $OPT $INC -c src/cudnn_real.cpp -o build/cudaemu/cudnn_real.o
$CXX -std=c++17 $OPT $INC -c src/cublas_real.cpp -o build/cudaemu/cublas_real.o
$CXX -std=c++17 $OPT $INC -c src/cudahost.cpp -o build/cudaemu/cudahost.o
echo "== the emulator"
$CXX -std=c++17 -O2 $INC -o "$OUT" src/vvcudaemu.cpp $EMU_SRC \
    build/cudaemu/cudakernels.o build/cudaemu/cudnn_real.o \
    build/cudaemu/cublas_real.o build/cudaemu/cudahost.o
ls -l "$OUT"
echo done
