#!/bin/sh
# Speak something the snapshot never heard.
#
# wslresume.sh checks that resuming reproduces the audio of a run that never
# stopped, which is the correctness question.  This is the useful one: the
# snapshot is taken with the sessions built and no kernel yet launched, so what
# the resumed run says is decided afterwards.  If it could only repeat the
# sentence it was saved with, this would be a cache of one answer rather than a
# warm start.
#
# What can vary is exactly what the guest reads *after* the snapshot point.  The
# command line is not that: argv was written into the guest's stack when it was
# loaded, and came back with the rest of its memory, so the resumed run is still
# working from the arguments the saved one had - including where it writes.  The
# text file is read in the step below the snapshot, so its contents are free.
set -e
cd "$(dirname "$0")/.."
OUT=sysroot/opt/vvcuda
SNAP=${VVSNAP:-$HOME/vv/session.state}
TEXT=${1:-こんにちは、げんきですか}

[ -f "$SNAP" ] || { echo "no $SNAP - run tools/wslresume.sh first"; exit 1; }
printf '%s' "$TEXT" > "$OUT/text.txt"
rm -f "$OUT/out.wav"

args="/opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 /opt/vvcuda/open_jtalk_dic_utf_8-1.11 /opt/vvcuda/0.vvm 3 @/opt/vvcuda/text.txt /opt/vvcuda/out.wav"

echo "== resume and say: $TEXT"
start=$(date +%s)
./vvcudaemu --sysroot "$PWD/sysroot" --resume "$SNAP" "$OUT/cudavvm" $args 2>&1 |
    grep -vE "^x86emu: open|^kernel " | tail -8
echo "elapsed $(( $(date +%s) - start )) s"
[ -f "$OUT/out.wav" ] || { echo "RESUME: no out.wav"; exit 1; }
python3 tools/wavstat.py "$OUT/out.wav"
