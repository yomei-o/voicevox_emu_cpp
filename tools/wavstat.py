"""Summarise a 16-bit PCM WAV: is there actually audio in it?

Where there is a reference, tools/wavcmp.py is the check.  Where there is not -
a text nothing has been run against before - this says whether what came out is
speech-shaped or silence, or the full-scale noise a broken kernel produces.

    python tools/wavstat.py out.wav
"""
import struct
import sys

sys.path.insert(0, __file__.rsplit("wavstat.py", 1)[0])
from wavcmp import read_wav  # noqa: E402  - same parser, one copy


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    fmt, s = read_wav(sys.argv[1])
    n = len(s)
    if not n:
        raise SystemExit("empty")
    peak = max(abs(v) for v in s)
    nonzero = sum(1 for v in s if v)
    energy = sum(float(v) * v for v in s) / n
    # How often the signal crosses zero, which separates speech from noise:
    # a vocoder's output crosses a few thousand times a second, white noise
    # around a quarter of the sample rate.
    crossings = sum(1 for i in range(1, n) if (s[i - 1] < 0) != (s[i] < 0))
    print("  %s" % sys.argv[1])
    print("  %d samples, %.2f s at %d Hz" % (n, n / fmt[2], fmt[2]))
    print("  peak            %d" % peak)
    print("  non-zero        %.1f %%" % (100.0 * nonzero / n))
    print("  rms             %.1f" % (energy ** 0.5))
    print("  zero crossings  %d/s" % (crossings * fmt[2] // n))


main()
