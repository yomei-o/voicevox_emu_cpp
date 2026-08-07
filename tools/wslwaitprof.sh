#!/bin/sh
# Block until the profiling run has written its report, then print it.
#
# A script rather than an inline loop: the pattern "^\[profile\]" does not
# survive the trip from a Windows shell through wsl.exe, and a grep whose
# pattern was mangled exits zero for the wrong reason.
LOG=$HOME/vvprof.log
while ! grep -q '^\[profile\]' "$LOG" 2>/dev/null; do
    pgrep -f vvcudaemu > /dev/null || {
        # The run is gone and there is no report: say so rather than spin.
        [ -f "$LOG" ] && tail -5 "$LOG"
        echo "no profile written - the run ended without one"
        exit 1
    }
    sleep 20
done
sh "$(dirname "$0")/wslproflog.sh"
