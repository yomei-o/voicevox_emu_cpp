#!/bin/sh
# How much AVX a binary contains, and which mnemonics.
#
# The emulator decodes SSE2 through SSE4.1 and no VEX prefix at all, so a
# library that *executes* AVX cannot run under it.  Containing AVX is not the
# same thing - ONNX Runtime's CPU kernels are full of it and pick a path from
# CPUID at run time, which is why the CPU build works here.  So the number to
# look at is not the total but whether the AVX sits in kernels behind a CPUID
# check or in ordinary compiler output that always runs.
#
#   sh tools/count_avx.sh <binary> [more...]
for f in "$@"; do
    [ -f "$f" ] || { echo "$f: not found"; continue; }
    objdump -d --no-show-raw-insn "$f" 2>/dev/null |
        grep -oP '^\s+[0-9a-f]+:\s+\Kv[a-z0-9]+' | sort | uniq -c | sort -rn > /tmp/avx_count.txt
    total=$(awk '{ s += $1 } END { print s + 0 }' /tmp/avx_count.txt)
    distinct=$(wc -l < /tmp/avx_count.txt)
    echo "$f"
    echo "  $total VEX instructions, $distinct distinct mnemonics"
    echo "  most common:"
    head -8 /tmp/avx_count.txt | sed 's/^/    /'
    echo "  the ones a compiler emits everywhere, not just in kernels:"
    grep -E ' (vmovq|vmovd|vzeroupper|vpxor|vmovdqa|vmovdqu|vmovss|vmovsd)$' /tmp/avx_count.txt |
        sed 's/^/    /' | head -10
    echo
done
