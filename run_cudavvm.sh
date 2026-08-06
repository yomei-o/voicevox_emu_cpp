#!/bin/sh
# The CUDA build of the runtime, under the emulator, against the shim.
#
# The whole shim story so far was measured natively, which was the right way
# round while the arithmetic was being got right - the runtime behaves the same
# either way and the loop is seconds instead of minutes.  This is the other
# half: does any of it *load* under x86emu, and what does it cost?
#
# Two things make this a real question rather than a formality.  The CUDA
# provider is 460 MB, and Memory::map allocates every page of a segment up
# front - so it is also 460 MB of guest pages before a single instruction runs.
# And the shim's arithmetic is, for now, still guest code: nothing here hooks it
# out to the host yet, so this measures the *loading*, not the speed.
#
#   VVCUDA=<dir with the CUDA build> sh run_cudavvm.sh ["text"]
#
# The provider wants tools/slim_provider.sh run over it first, or the 419 MB of
# GPU machine code nobody executes gets paged in too.
set -e
cd "$(dirname "$0")"
. "$(dirname "$0")/emu_path.sh"

VVCUDA=${VVCUDA:-cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib}
VVM=${VVM:-0.vvm}
TEXT=${1:-あ}
OUT=sysroot/opt/vvcuda

[ -f "$VVCUDA/libvoicevox_onnxruntime.so.1.17.3" ] || {
    echo "no CUDA build at $VVCUDA - set VVCUDA="; exit 1; }

sh "$(dirname "$0")/unpack.sh"
mkdir -p "$OUT"
cp guest/cudavvm guest/libvoicevox_core.so "guest/$VVM" "$OUT/"

# The stand-ins go where a Debian with CUDA installed would keep them, not on
# LD_LIBRARY_PATH.  Two reasons, and the second is the one that bit: the
# provider's own dependencies are resolved against *its* RUNPATH and the system
# directories, and MSYS rewrites any variable whose name ends in PATH before a
# native Windows program ever sees it - so LD_LIBRARY_PATH=/opt/vvcuda reached
# the guest as a Windows path and did nothing at all.
#
# And they must be built for the *guest*, which is SSE2: the default flags aim
# at the host and include -mavx2, and an emulator with no VEX decoder stops on
# the first `vmovq` of our own stub.  OPT=-O2 is what makes them guest code.
mkdir -p sysroot/lib/x86_64-linux-gnu
cp guest/cudastub/*.so.* sysroot/lib/x86_64-linux-gnu/
cp "$VVCUDA/libvoicevox_onnxruntime.so.1.17.3" "$OUT/"
cp "$VVCUDA/libvoicevox_onnxruntime_providers_shared.so" "$OUT/"

# The big one.  Prefer a slimmed copy if there is one beside the original.
slim="$VVCUDA/libvoicevox_onnxruntime_providers_cuda_slim.so"
[ -f "$slim" ] || slim="$VVCUDA/libvoicevox_onnxruntime_providers_cuda.so"
cp "$slim" "$OUT/libvoicevox_onnxruntime_providers_cuda.so"
echo "provider: $slim ($(wc -c < "$slim") bytes)"

[ -d "$OUT/open_jtalk_dic_utf_8-1.11" ] || cp -r guest/open_jtalk_dic_utf_8-1.11 "$OUT/"

# The runtime opens its providers by soname from wherever the loader looks, and
# the guest has no /usr/local/cuda: LD_LIBRARY_PATH is what makes the stand-ins
# findable.
# The text goes in through a file, not the command line: on Windows the
# argument arrives re-encoded and the CORE rejects it as invalid UTF-8 - which
# is the right answer to the wrong bytes.
printf '%s' "$TEXT" > "$OUT/text.txt"

export MSYS2_ARG_CONV_EXCL='*' MSYS_NO_PATHCONV=1
exec "$EMU" --sysroot sysroot \
    sysroot/opt/vvcuda/cudavvm \
    /opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 \
    /opt/vvcuda/open_jtalk_dic_utf_8-1.11 \
    "/opt/vvcuda/$VVM" 3 @/opt/vvcuda/text.txt /opt/vvcuda/out.wav
