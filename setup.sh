#!/bin/sh
# One command to get from a fresh clone to a runnable tree.
#
#   sh setup.sh          # emulator, sysroot, models, guests
#   sh run_probe.sh      # does the runtime come up?      (~10 s)
#   sh run_tts.sh        # the whole thing                (slow, see resume.md)
#
# Nothing fetched here is in the repository.  The voice models in particular
# are covered by terms that forbid redistribution, so they are downloaded from
# VOICEVOX rather than committed; the same script keeps the runtime, the
# dictionary and a 3.4 MB slice of Debian out of the clone as a side effect.
set -e
cd "$(dirname "$0")"

echo "=== 1/4  emulator"
if [ -x ../x86_emu_cpp/x86emu.exe ] || [ -x ../x86_emu_cpp/x86emu ]; then
    echo "    using the sibling checkout at ../x86_emu_cpp"
else
    (cd x86_emu_cpp && sh build.sh)
fi

echo "=== 2/4  sysroot (Debian bookworm amd64)"
sh make_sysroot.sh

echo "=== 3/4  runtime, core, voice model, dictionary"
sh fetch_models.sh "${VVM:-0.vvm}"

echo "=== 4/4  guests"
sh build.sh

echo
echo "ready.  sh run_probe.sh"
