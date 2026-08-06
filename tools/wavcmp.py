"""Compare two 16-bit PCM WAV files sample for sample.

The same job as tools/wavcmp.mjs, for machines without node.  Prints the
largest difference against the peak, which is the number this project has been
quoting all along: "the same utterance, through four very different machines".

    python tools/wavcmp.py a.wav b.wav
"""
import struct
import sys


def read_wav(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SystemExit("%s: not a RIFF/WAVE file" % path)
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body, 0)
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise SystemExit("%s: missing fmt or data chunk" % path)
    if fmt[5] != 16:
        raise SystemExit("%s: %d-bit, expected 16" % (path, fmt[5]))
    n = len(pcm) // 2
    return fmt, struct.unpack_from("<%dh" % n, pcm, 0)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    fa, a = read_wav(sys.argv[1])
    fb, b = read_wav(sys.argv[2])
    print("  %-28s %d ch, %d Hz, %d samples" % (sys.argv[1], fa[1], fa[2], len(a)))
    print("  %-28s %d ch, %d Hz, %d samples" % (sys.argv[2], fb[1], fb[2], len(b)))
    if fa[1] != fb[1] or fa[2] != fb[2]:
        print("  channel count or rate differs")
    n = min(len(a), len(b))
    if len(a) != len(b):
        print("  lengths differ by %d samples; comparing the first %d"
              % (abs(len(a) - len(b)), n))
    worst, at, differing = 0, -1, 0
    for i in range(n):
        d = abs(a[i] - b[i])
        if d:
            differing += 1
        if d > worst:
            worst, at = d, i
    peak = max(max((abs(v) for v in a[:n]), default=0),
               max((abs(v) for v in b[:n]), default=0))
    print("")
    print("  samples differing   %d of %d" % (differing, n))
    print("  largest difference  %d at sample %d" % (worst, at))
    print("  peak amplitude      %d" % peak)
    if peak:
        print("  worst as a fraction %.4f %%" % (100.0 * worst / peak))
    sys.exit(0 if worst == 0 else 1)


main()
