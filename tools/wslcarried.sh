#!/bin/sh
# The carried CUDA runtime, against the sixteen reference values.
set -e
cd "$(dirname "$0")/.."
NODE=$(ls "$HOME"/emsdk/node/*/bin/node 2>/dev/null | head -1)
[ -x "$NODE" ] || { echo "no node in emsdk"; exit 1; }
"$NODE" web/test_carried.mjs 2>&1 | sed -n "s/^ *\[[0-9]*\] //p" > build/carried.txt
grep -v "^#" colab/predict_duration_reference.txt | grep . > build/ref.txt
paste build/carried.txt build/ref.txt | awk "{ d = \$1 - \$2; if (d < 0) d = -d; if (d >= 1e-5) bad++ } END { if (NR == 0) print \"no values came back\"; else if (bad) printf \"%d of %d disagree\n\", bad, NR; else printf \"all %d values match\n\", NR }"
