#!/bin/sh
# The first kernel whose output differs, with a few on either side for context.
cd "$(dirname "$0")/.."
n=$(diff --unchanged-line-format='' --old-line-format='%dn
' --new-line-format='' \
        build/k_native.txt build/k_emu.txt 2>/dev/null | head -1)
[ -n "$n" ] || { echo "no difference"; exit 0; }
echo "first difference at line $n"
echo
echo "== native"
sed -n "$((n > 3 ? n - 3 : 1)),$((n + 2))p" build/k_native.txt
echo
echo "== emulated"
sed -n "$((n > 3 ? n - 3 : 1)),$((n + 2))p" build/k_emu.txt
