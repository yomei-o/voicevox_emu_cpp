#!/bin/sh
# The release WebAssembly build against the sixteen reference values.
#
# The debug build agreeing with the native shim is not the same as the release
# build being right: the two differed before, one crashing where the other
# returned wrong numbers.  This runs the -O3 module and holds it to
# colab/predict_duration_reference.txt - the values a desktop CPU and a Tesla T4
# both produce.
set -e
cd "$(dirname "$0")/.."
EMCC=$HOME/emsdk/upstream/emscripten/emcc
NODE=$(ls "$HOME"/emsdk/node/*/bin/node | head -1)

EMCC="$EMCC" sh web/build_cuda.sh > /dev/null
echo "== cudaprobe, release build"
"$NODE" web/test_cuda_probe.mjs 2>&1 | sed -n 's/^ *\[[0-9]*\] //p' > build/wasm_probe.txt
grep -v '^#' colab/predict_duration_reference.txt | grep . > build/ref.txt
paste build/wasm_probe.txt build/ref.txt | awk '
    { d = $1 - $2; if (d < 0) d = -d
      status = (d < 1e-5) ? "ok" : sprintf("DIFF %.6f", d)
      if (d >= 1e-5) bad++
      printf "  [%2d] %-10s %-10s %s\n", NR - 1, $1, $2, status }
    END { print ""
          if (NR == 0) print "no values came back"
          else if (bad) printf "%d of %d values disagree\n", bad, NR
          else printf "all %d values match\n", NR }'
