#!/bin/sh
# The TTS regression, detached, writing to a log.
#
#   sh tools/wslregress.sh          start it, print the log's path, return
#   tail -f build/regress.log       watch it
#
# It takes about seventy minutes, so the point of the log is to be able to see
# where it has got to without waiting for the end.
#
# Two ways of running it have now failed, both silently:
#
#   through `| tail`, so nothing appeared until it finished - and when it was
#   cut short at ten minutes there was nothing at all, which reads as a crash
#
#   with nohup, backgrounded inside `wsl.exe -- sh ...` - the WSL session tears
#   down when wsl.exe returns and takes the job with it, leaving an empty log
#
# So it runs in the foreground here and the caller keeps wsl.exe alive for as
# long as it takes.  `tee` puts it on the log and on stdout both.
set -e
cd "$(dirname "$0")/.."
mkdir -p build
LOG=$PWD/build/regress.log
: > "$LOG"
echo "logging to $LOG"
sh tools/regress_tts.sh 2>&1 | tee -a "$LOG"
