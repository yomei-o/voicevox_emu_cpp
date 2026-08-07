#!/bin/sh
# Why the snapshot did not happen: is the call in the binary, and does the
# variable reach the guest?
cd "$(dirname "$0")/.."
echo "== is the call compiled in?"
strings guest/cudavvm | grep -c VVSNAPSHOT
strings guest/cudavvm | grep "snapshot" | head -3
echo "== is it in the staged copy?"
strings sysroot/opt/vvcuda/cudavvm 2>/dev/null | grep -c VVSNAPSHOT
echo "== does the guest see the environment at all?"
VVSNAPSHOT=/tmp/x printf '%s\n' "host sees: $VVSNAPSHOT"
