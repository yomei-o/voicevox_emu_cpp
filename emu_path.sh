# Sourced by the run scripts: picks the emulator and says which one.
#
# The vendored copy wins.  It used to be the other way round, on the theory that
# a machine with both checkouts was developing in the sibling - but the fixes
# this project needs live *here* until they are merged back, and a sibling built
# before them fails in exactly the way this project spent a day explaining.
# Silently running the wrong emulator is the worst outcome available, so the
# choice is printed.
#
# EMU= overrides everything.
if [ -z "$EMU" ]; then
    for cand in x86_emu_cpp/build/Release/x86emu.exe \
                x86_emu_cpp/x86emu.exe x86_emu_cpp/x86emu \
                ../x86_emu_cpp/x86emu.exe ../x86_emu_cpp/x86emu; do
        [ -x "$cand" ] && { EMU=$cand; break; }
    done
fi
[ -n "$EMU" ] || { echo "no emulator: run setup.sh, or set EMU="; exit 1; }
case "$EMU" in
    ../*) echo "emulator: $EMU  (the sibling checkout - does it have this project's fixes?)" ;;
    *) echo "emulator: $EMU" ;;
esac

# Older than the sources it was built from?
#
# Printing the path was supposed to be enough and is not: the path is right and
# the binary behind it is a week old.  A seventy-minute regression ran twice
# against an emulator built before the day's changes, and reported on none of
# them.  Nothing about that run looked wrong.
newer=$(find x86_emu_cpp/src -newer "$EMU" -name '*.cpp' -o -newer "$EMU" -name '*.h' 2>/dev/null | head -3)
if [ -n "$newer" ]; then
    echo "  STALE: these are newer than the emulator -"
    printf '    %s\n' $newer
    echo "  rebuild it (cmake --build x86_emu_cpp/build --config Release), or set EMU= on purpose."
    exit 1
fi
