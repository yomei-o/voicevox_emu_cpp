#!/bin/sh
# Build the sessions once, then speak from the snapshot.
#
# The whole point of a snapshot here is the 250 seconds that go into decrypting
# and building the sessions.  This takes one run as far as that and stops, then
# resumes from the file and synthesises - and the audio has to be the same audio
# the run that never stopped produced.
set -e
cd "$(dirname "$0")/.."
OUT=sysroot/opt/vvcuda
SNAP=${VVSNAP:-$HOME/vv/session.state}

sh unpack.sh > /dev/null
mkdir -p "$OUT" sysroot/lib/x86_64-linux-gnu
cp guest/cudavvm guest/libvoicevox_core.so guest/0.vvm "$OUT/"
cp guest/cudaguest/*.so.* sysroot/lib/x86_64-linux-gnu/
CUDA=${VVCUDA:-$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib}
cp "$CUDA/libvoicevox_onnxruntime.so.1.17.3" "$CUDA/libvoicevox_onnxruntime_providers_shared.so" "$OUT/"
slim=$HOME/vv/slimtest/libvoicevox_onnxruntime_providers_cuda.so
[ -f "$slim" ] || slim="$CUDA/libvoicevox_onnxruntime_providers_cuda.so"
cp "$slim" "$OUT/libvoicevox_onnxruntime_providers_cuda.so"
[ -d "$OUT/open_jtalk_dic_utf_8-1.11" ] || cp -r guest/open_jtalk_dic_utf_8-1.11 "$OUT/"
printf '%s' "ずんだもんなのだ" > "$OUT/text.txt"

args="/opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3 /opt/vvcuda/open_jtalk_dic_utf_8-1.11 /opt/vvcuda/0.vvm 3 @/opt/vvcuda/text.txt /opt/vvcuda/out.wav"

echo "== build the sessions and stop"
rm -f "$SNAP" "$SNAP.shim" "$OUT/out.wav"
start=$(date +%s)
./vvcudaemu --sysroot "$PWD/sysroot" --env "VVSNAPSHOT=$SNAP" "$OUT/cudavvm" $args 2>&1 |
    grep -vE "^x86emu: open|^kernel " | tail -6
echo "elapsed $(( $(date +%s) - start )) s"
ls -l "$SNAP" "$SNAP.shim" 2>&1 | sed 's/^/  /'

echo
echo "== resume and speak"
start=$(date +%s)
./vvcudaemu --sysroot "$PWD/sysroot" --resume "$SNAP" "$OUT/cudavvm" $args 2>&1 |
    grep -vE "^x86emu: open|^kernel " | tail -10
echo "elapsed $(( $(date +%s) - start )) s"

[ -f "$OUT/out.wav" ] || { echo "RESUME: no out.wav"; exit 1; }
python3 tools/wavcmp.py "$OUT/out.wav" web/sample/gpu_zundamon.wav
