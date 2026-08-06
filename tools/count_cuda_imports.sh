#!/bin/sh
# How much CUDA would a shim have to implement?
#
# The idea it answers: run the *CUDA* build of voicevox_onnxruntime inside the
# emulator and hook its calls into libcudart/cuBLAS/cuDNN, doing the maths
# natively.  That boundary is the only visible one - the CPU build exports
# exactly three symbols, so there is nothing inside it to hook - so the question
# is just how wide the boundary is.
#
# Run it on a Linux x86-64 machine against the extracted CUDA release:
#
#   sh tools/count_cuda_imports.sh <dir with libvoicevox_onnxruntime_providers_cuda.so>
LIB=${1:-.}/libvoicevox_onnxruntime_providers_cuda.so
[ -f "$LIB" ] || { echo "no $LIB"; exit 1; }

echo "== undefined symbols by library"
for pfx in cuda cublas cudnn cufft curand cusparse nvrtc nvToolsExt; do
    n=$(nm -D --undefined-only "$LIB" | grep -c "U ${pfx}")
    printf '%-12s %s\n' "$pfx" "$n"
done
echo
printf 'total undefined  %s\n' "$(nm -D --undefined-only "$LIB" | wc -l)"

echo
echo "== the cudart entry points it actually calls"
nm -D --undefined-only "$LIB" | awk '{print $2}' | grep '^cuda' | sort -u

echo
echo "== cuBLAS / cuDNN entry points"
nm -D --undefined-only "$LIB" | awk '{print $2}' | grep -E '^(cublas|cudnn)' | sort -u | head -60

echo
echo "== how many of its own kernels it registers"
# Every kernel a CUDA binary launches is registered by name at load time; the
# names live in the .nv_fatbin / .rodata.  This counts the distinct ones, which
# is what a shim would have to reimplement on top of the API above.
strings -a "$LIB" | grep -cE '^_ZN11onnxruntime4cuda' || true
