#!/bin/sh
# Run the CUDA shim against predict_duration.onnx and compare with the reference.
#
# The reference is what the CPU provider on a desktop and the CUDA provider on a
# Colab T4 both answer, to every digit printed - so a disagreement here is the
# shim's, and the index of the first one says where to look.
#
#   sh tools/check_shim.sh [dir with cudaprobe and the libraries]
set -e
cd "$(dirname "$0")/.."
REF="$PWD/colab/predict_duration_reference.txt"
RUN=${1:-$HOME/vv/cudarun}
cd "$RUN"

LD_LIBRARY_PATH=. ./cudaprobe ./libvoicevox_onnxruntime.so.1.17.3 \
    ./predict_duration.onnx 2>/dev/null |
    sed -n 's/^ *\[[0-9]*\] //p' > /tmp/shim_got.txt

grep -v '^#' "$REF" | grep . > /tmp/shim_ref.txt

paste /tmp/shim_got.txt /tmp/shim_ref.txt | awk '
    { d = $1 - $2; if (d < 0) d = -d
      status = (d < 1e-5) ? "ok" : sprintf("DIFF %.6f", d)
      if (d >= 1e-5) bad++
      printf "  [%2d] %-10s %-10s %s\n", NR - 1, $1, $2, status }
    END { print ""
          if (bad) printf "%d of %d values disagree\n", bad, NR
          else     printf "all %d values match\n", NR }'
