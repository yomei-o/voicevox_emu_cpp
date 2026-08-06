#!/bin/sh
# Empty the GPU machine code out of libvoicevox_onnxruntime_providers_cuda.so.
#
# The provider is 440 MB and 419 MB of that is `.nv_fatbin` - the compiled
# device code for all 4757 kernels.  The shim never executes one instruction of
# it: `__cudaRegisterFatBinary` gets a pointer that is only ever handed back,
# and the kernel *names* that `__cudaRegisterFunction` reports live in .rodata,
# not in the fatbin.  So the bytes are dead weight, and this writes zeros over
# them.
#
# The section stays where it is and keeps its size, so every offset in the file
# is still correct and the loader is none the wiser.  What changes is that 419
# MB of zeros compress to nothing:
#
#     440 MB  ->  9.3 MB gzipped
#
# Verified: the provider still registers 4757 kernels, still launches 377 for an
# utterance, and the audio is unchanged - within 2 of 12988 of the Tesla T4.
#
# This is what makes the browser arithmetic worth considering at all; a
# half-gigabyte download is not something a demo page can ask for.  It does
# *not* fix the memory side: the emulator's Memory::map allocates every page of
# a segment up front and then writes the whole thing, so the zeros still cost
# 419 MB of guest pages.  Lazy pages would fix that and would help everything
# else too.
#
#   sh tools/slim_provider.sh <in.so> [out.so]
set -e
IN=$1
OUT=${2:-${IN%.so}_slim.so}
[ -f "$IN" ] || { echo "usage: sh tools/slim_provider.sh <providers_cuda.so> [out.so]"; exit 1; }

read -r off size <<EOF
$(readelf -S -W "$IN" | awk '$2 == ".nv_fatbin" { print $5, $6 }')
EOF
[ -n "$size" ] || { echo "no .nv_fatbin in $IN - nothing to do"; exit 1; }

cp "$IN" "$OUT"
python3 - "$OUT" "$off" "$size" <<'PY'
import sys
path, off, size = sys.argv[1], int(sys.argv[2], 16), int(sys.argv[3], 16)
with open(path, "r+b") as f:
    f.seek(off)
    left, chunk = size, 1 << 20
    while left:
        n = min(chunk, left)
        f.write(bytes(n))
        left -= n
print("zeroed %d bytes at 0x%x" % (size, off))
PY

before=$(wc -c < "$IN")
after=$(gzip -9 -c "$OUT" | wc -c)
echo "$IN  $before bytes"
echo "$OUT  $before bytes, $after gzipped"
