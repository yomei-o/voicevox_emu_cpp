#!/bin/sh
# A long utterance through the shim, to see whether anything goes unhandled.
#
# The kernel table was written against two short utterances, and some of ONNX
# Runtime's kernels are specialised on *size*: softmax_warp_forward takes
# log2(element count) as a template parameter, _SliceKernel takes the rank.
# Those specialisations are matched by prefix here and the implementations read
# the real counts from their arguments, so they should hold - but "should" is
# not a measurement, and an unimplemented launch now says so at the end.
#
# There is no reference audio for this text, so what is being checked is the
# warning count, not the samples.
cd "$(dirname "$0")/.."
rm -f sysroot/opt/vvcuda/out.wav
sh tools/wslrun_cuda.sh "ずんだもんなのだ。きょうはとてもいいてんきなので、こうえんまでさんぽにいってきました。とちゅうでねこにあったのだ。" 2>&1 |
    grep -vE "^x86emu: open" |
    grep -E "tts took|produced|WARNING|launches|^kernel " | sort | uniq -c | sort -rn | head -15
[ -f sysroot/opt/vvcuda/out.wav ] && ls -l sysroot/opt/vvcuda/out.wav
