#!/bin/sh
# How big is the state after the sessions are built, and how much of it
# compresses?
#
# Building the sessions is minutes and synthesis is seconds, so the obvious
# thing to want is to do the first once and resume from it.  Whether that is
# practical turns on a number nobody had.  This writes the state and measures
# it; restoring is the other half and is not written yet.
#
# The file goes to $HOME, not into the repository, and it is not something to
# publish: it contains the decrypted voice model, in guest memory and again in
# the device arena.
set -e
cd "$(dirname "$0")/.."
SNAP=$HOME/vvsession.snap

echo "== rebuilding the emulator (it carries the --env the guest needs)"
sh tools/wslbuild.sh > /dev/null 2>&1 || { sh tools/wslbuild.sh; exit 1; }

echo "== rebuilding the guest program (it makes the call)"
( cd "$HOME/vv/cudarun" &&
  gcc -O2 -I/mnt/c/prog/claude/voicevox_emu_cpp/src -o cudavvm \
      /mnt/c/prog/claude/voicevox_emu_cpp/src/cudavvm.c \
      -L. -l:libvoicevox_core.so -ldl )
cp "$HOME/vv/cudarun/cudavvm" guest/cudavvm

echo "== running to the point a resume would start from"
rm -f "$SNAP" sysroot/opt/vvcuda/out.wav
VVSNAPSHOT=$SNAP sh tools/wslrun_cuda.sh "ずんだもんなのだ" > "$HOME/vvsnap.log" 2>&1 || true
grep -vE "^x86emu: open|^kernel " "$HOME/vvsnap.log" |
    grep -E "snapshot|\[snap\]|tts took|produced|WARNING"

[ -f "$SNAP" ] || {
    echo "no snapshot written - the last of the run was:"
    grep -vE "^x86emu: open|^kernel " "$HOME/vvsnap.log" | tail -8
    exit 1
}
raw=$(wc -c < "$SNAP")
echo
echo "== how it compresses"
printf '  %-10s %12d bytes  %6.1f MB\n' raw "$raw" "$(echo "$raw" | awk '{print $1/1048576}')"
for m in "gzip -9" "xz -9 -T0" "zstd -19 -T0 -c"; do
    tool=${m%% *}
    command -v "$tool" > /dev/null 2>&1 || { echo "  $tool: not installed"; continue; }
    n=$($m -c "$SNAP" 2>/dev/null | wc -c)
    printf '  %-10s %12d bytes  %6.1f MB   %.1f %% of raw\n' "$tool" "$n" \
        "$(echo "$n" | awk '{print $1/1048576}')" \
        "$(echo "$n $raw" | awk '{print 100*$1/$2}')"
done
