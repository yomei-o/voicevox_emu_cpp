#!/bin/sh
# The full emulated pipeline, end to end, against the audio it produced before.
#
# This is the regression check for changes to the emulator core.  It is slow -
# the CPU build's vocoder interpreted is the better part of an hour - so it is
# the one to start and come back to, not the one to run in a loop.
#
# It deletes the previous out.wav first, deliberately.  A run that is cut short
# leaves the old file in place, and comparing *that* against the reference it
# was copied from reports zero difference and means nothing.
set -e
cd "$(dirname "$0")/.."
rm -f sysroot/opt/vv/out.wav
start=$(date +%s)
sh run_tts.sh
echo "elapsed $(( $(date +%s) - start )) s"
[ -f sysroot/opt/vv/out.wav ] || { echo "REGRESS: no out.wav - the run did not finish"; exit 1; }
python tools/wavcmp.py sysroot/opt/vv/out.wav web/sample/emu_zundamon.wav
