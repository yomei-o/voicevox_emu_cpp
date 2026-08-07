"""Break a session snapshot down, and see what would make it smaller.

The whole thing compresses to about 127 MB with xz, and the question on the
table is whether it fits in 100.  Two things decide that: which half is the
bulk, and whether float weights compress better when their bytes are
de-interleaved.

Byte shuffling is the standard trick for arrays of floats.  Consecutive
float32s share their exponents and differ in their low mantissa bytes, so a
compressor sees a repeating pattern smeared across four-byte strides; splitting
the stream into four planes by byte position puts the similar bytes next to
each other.  It costs nothing to undo.

    python3 tools/snapanalyse.py <snapshot>
"""
import lzma
import struct
import subprocess
import sys
import zlib

PAGE = 4096


def read_snapshot(path):
    with open(path, "rb") as f:
        data = f.read()
    # Pages first: (index u64, 4096 bytes) each.  The arena's length is the
    # first u64 that cannot be a plausible continuation, so instead of guessing,
    # walk from the end: the tail is (arena_len u64, arena_len bytes).
    n = len(data)
    # Try each possible page count: the arena length sits right after them.
    for pages in range((n - 8) // (8 + PAGE) + 1, -1, -1):
        off = pages * (8 + PAGE)
        if off + 8 > n:
            continue
        arena_len = struct.unpack_from("<Q", data, off)[0]
        if off + 8 + arena_len == n:
            return data[:off], data[off + 8:], pages
    raise SystemExit("cannot parse %s" % path)


def squeeze(name, blob):
    if not blob:
        return
    xz = len(lzma.compress(blob, preset=9 | lzma.PRESET_EXTREME))
    gz = len(zlib.compress(blob, 9))
    print("  %-28s %7.1f MB raw -> %6.1f gzip, %6.1f xz  (%.1f %%)"
          % (name, len(blob) / 1048576, gz / 1048576, xz / 1048576,
             100.0 * xz / len(blob)))
    return xz


def shuffle4(blob):
    """Split into four planes by byte position: the float trick."""
    n = len(blob) - len(blob) % 4
    planes = [bytearray(n // 4) for _ in range(4)]
    for k in range(4):
        planes[k][:] = blob[k:n:4]
    return b"".join(bytes(p) for p in planes) + blob[n:]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    pages_blob, arena, pages = read_snapshot(sys.argv[1])
    print("  %d pages (%.1f MB with their indices), arena %.1f MB"
          % (pages, len(pages_blob) / 1048576, len(arena) / 1048576))
    print()
    # The page bodies without their 8-byte indices, so the two halves compare.
    bodies = b"".join(pages_blob[i * (8 + PAGE) + 8:(i + 1) * (8 + PAGE)]
                      for i in range(pages))
    a = squeeze("guest pages", bodies)
    b = squeeze("arena", arena)
    c = squeeze("arena, bytes de-interleaved", shuffle4(arena))
    d = squeeze("guest pages, de-interleaved", shuffle4(bodies))
    print()
    best_guest = min(x for x in (a, d) if x)
    best_arena = min(x for x in (b, c) if x)
    print("  best of each, together: %.1f MB" % ((best_guest + best_arena) / 1048576))


main()
