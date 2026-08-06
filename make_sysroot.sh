#!/bin/sh
# Builds sysroot/ - a Debian bookworm amd64 filesystem root, enough to load and
# run a dynamically linked C++ program.  No Linux host and no WSL: these are
# Debian's own packages, unpacked with ar and tar.
#
# Two things a naive unpack gets wrong, both of which cost a debugging session:
#
#   - Windows cannot create the symlinks the .debs carry, so every one is
#     materialised as a copy of its target.  A dangling symlink presents to
#     ld.so as "cannot open shared object file: Invalid argument" on a file
#     that is plainly there.
#   - Debian leaves the SONAME links (libfoo.so.1 -> libfoo.so.1.2.3) to
#     ldconfig at install time.  They are not in the archive; make them here.
set -e
cd "$(dirname "$0")"

ROOT=sysroot
MIRROR=http://deb.debian.org/debian/pool/main

# glibc 2.36: libvoicevox_onnxruntime wants 2.29, libvoicevox_core wants 2.34.
GLIBC=2.36-9+deb12u14
GCC=12.2.0-14+deb12u1

PKGS="
$MIRROR/g/glibc/libc6_${GLIBC}_amd64.deb
$MIRROR/g/glibc/libc6-dev_${GLIBC}_amd64.deb
$MIRROR/g/gcc-12/libgcc-s1_${GCC}_amd64.deb
$MIRROR/g/gcc-12/libstdc++6_${GCC}_amd64.deb
$MIRROR/g/gcc-12/libgcc-12-dev_${GCC}_amd64.deb
"

mkdir -p debs "$ROOT"
for url in $PKGS; do
    f=debs/$(basename "$url")
    [ -f "$f" ] || { echo "fetch $(basename "$url")"; curl -sL -o "$f" "$url"; }
done

: > debs/links.txt
for d in debs/*.deb; do
    (cd debs && rm -f data.tar.* control.tar.* debian-binary && ar x "$(basename "$d")")
    df=$(ls debs/data.tar.* | head -1)
    # Record the symlinks before extracting, because extraction drops them.
    tar tvf "$df" | grep '^l' | sed -E 's/.* ([^ ]+) -> (.*)$/\1\t\2/' >> debs/links.txt
    tar xf "$df" -C "$ROOT" 2>/dev/null || true
done
rm -f debs/data.tar.* debs/control.tar.* debs/debian-binary

echo "== materialising symlinks as copies"
n=0
while IFS="$(printf '\t')" read -r src tgt; do
    s="$ROOT/${src#./}"
    dir=$(dirname "$s")
    case "$tgt" in
        /*) t="$ROOT$tgt" ;;
        *)  t="$dir/$tgt" ;;
    esac
    if [ -f "$t" ] && [ ! -e "$s" ]; then mkdir -p "$dir"; cp "$t" "$s"; n=$((n + 1)); fi
done < debs/links.txt
echo "   $n copies"

echo "== creating SONAME names ldconfig would have made"
m=0
for f in $(find "$ROOT" -name '*.so.*' -type f); do
    so=$(objdump -p "$f" 2>/dev/null | awk '/SONAME/{print $2}')
    [ -n "$so" ] || continue
    dir=$(dirname "$f")
    [ -e "$dir/$so" ] || { cp "$f" "$dir/$so"; m=$((m + 1)); }
done
echo "   $m copies"

# The ELF interpreter every guest names in its PT_INTERP.
mkdir -p "$ROOT/lib64"
[ -e "$ROOT/lib64/ld-linux-x86-64.so.2" ] ||
    cp "$ROOT/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2" "$ROOT/lib64/"

# Somewhere for the guest to put a temporary file.  Open JTalk compiles a user
# dictionary through one, and voicevox_open_jtalk_rc_use_user_dict fails with
# USE_USER_DICT_ERROR when there is nowhere to put it.
mkdir -p "$ROOT/tmp"

# ORT's cpuinfo parses this and complains to stderr when it is absent.  The
# flags are the ones the emulator's CPUID actually advertises.
mkdir -p "$ROOT/proc"
cat > "$ROOT/proc/cpuinfo" <<'EOF'
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 42
model name	: x86emu
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2
cache size	: 256 KB
siblings	: 1
cpu cores	: 1

EOF

echo
for p in lib64/ld-linux-x86-64.so.2 lib/x86_64-linux-gnu/libc.so.6 \
         usr/lib/x86_64-linux-gnu/libstdc++.so.6 lib/x86_64-linux-gnu/libgcc_s.so.1 \
         usr/lib/x86_64-linux-gnu/crt1.o usr/lib/gcc/x86_64-linux-gnu/12/crtbegin.o \
         usr/include/dlfcn.h; do
    [ -e "$ROOT/$p" ] && echo "ok      $p" || { echo "MISSING $p"; fail=1; }
done
[ -n "$fail" ] && exit 1
echo
echo "sysroot ready"
