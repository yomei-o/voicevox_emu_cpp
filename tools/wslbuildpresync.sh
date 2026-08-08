#!/bin/sh
# Build vvcudaemu from the sources as they were before today's upstream sync,
# so the sync itself can be measured rather than assumed.
#
# The five files are kept in build/presync (put there straight from git), swapped
# in, built, and swapped back.  The working tree is restored even if the build
# fails - which is the only reason this is a script and not three commands.
set -e
cd "$(dirname "$0")/.."
[ -d build/presync ] || { echo "no build/presync - see resume.md"; exit 1; }

restore() {
    for f in build/presync/*; do
        cp "build/current/$(basename "$f")" "x86_emu_cpp/src/$(basename "$f")"
    done
    echo "  working tree restored"
}

mkdir -p build/current
for f in build/presync/*; do
    cp "x86_emu_cpp/src/$(basename "$f")" "build/current/$(basename "$f")"
done
trap restore EXIT

for f in build/presync/*; do
    cp "$f" "x86_emu_cpp/src/$(basename "$f")"
done
sh tools/wslbuild.sh > "$HOME/presync_build.log" 2>&1 || {
    echo "build failed - see $HOME/presync_build.log"
    tail -20 "$HOME/presync_build.log"
    exit 1
}
cp vvcudaemu vvcudaemu.presync
echo "  built ./vvcudaemu.presync"
