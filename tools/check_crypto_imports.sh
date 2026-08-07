#!/bin/sh
# Does the runtime import its cryptography, or carry it?
#
# This reads the dynamic symbol table - the same thing ld.so reads, and the same
# technique tools/count_cuda_imports.sh uses on the CUDA provider.  It is not an
# examination of *how* anything works: an imported symbol is a declaration the
# binary makes about itself.
#
# The question it answers is whether there is a seam.  A cipher that arrives
# through the PLT could be given a faster implementation the way libcudart was.
# One that is compiled in has no seam, and finding one would mean disassembling
# the protection, which is the thing the voice model's terms forbid - so a null
# result here is the end of the road, not the start of a harder search.
#
#   sh tools/check_crypto_imports.sh <library> [...]
for f in "$@"; do
    [ -f "$f" ] || { echo "$f: not found"; continue; }
    echo "== $f"
    total=$(nm -D --undefined-only "$f" 2>/dev/null | wc -l)
    echo "   $total undefined symbols in all"
    hits=$(nm -D --undefined-only "$f" 2>/dev/null | awk '{print $NF}' |
           grep -iE 'crypt|aes|cipher|evp_|ssl|sha[0-9]|md5|hmac|rand|chacha|salsa|blake|nettle|gcry|sodium|mbedtls|bcrypt' |
           sort -u)
    if [ -n "$hits" ]; then
        echo "   possibly cryptographic:"
        echo "$hits" | sed 's/^/     /'
    else
        echo "   nothing that looks cryptographic is imported"
    fi
    echo "   libraries it needs:"
    objdump -p "$f" 2>/dev/null | awk '/NEEDED/ { print "     " $2 }'
    echo
done
