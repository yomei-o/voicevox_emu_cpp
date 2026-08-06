# Sourced by the run scripts.  Prefers a sibling x86_emu_cpp checkout - on the
# machine that has both, that one is the source of truth - and falls back to
# the copy vendored here, which is what a fresh clone builds.
if [ -z "$EMU" ]; then
    for cand in ../x86_emu_cpp/x86emu.exe ../x86_emu_cpp/x86emu \
                x86_emu_cpp/x86emu.exe x86_emu_cpp/x86emu; do
        [ -x "$cand" ] && { EMU=$cand; break; }
    done
fi
[ -n "$EMU" ] || { echo "no emulator: run setup.sh, or set EMU="; exit 1; }
