#!/bin/sh
# Stands in for the emulator so the host half of the API can be developed and
# tested without it.
#
# vvhost.c starts the backend as `<emulator> --sysroot <dir> <program>`.  On a
# Linux x86-64 machine the guest agent runs natively, so this takes the same
# command line, drops the two emulator arguments and runs the program directly.
# A round trip then costs milliseconds instead of hours, which is the only way
# the protocol and the 63 entry points are worth debugging.
#
#     VOICEVOX_EMU_EMULATOR=./nativeemu.sh \
#     VOICEVOX_EMU_SYSROOT=<dir holding opt/vv/vvagent> ./yourprogram
shift 2
exec "$@"
