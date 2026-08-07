#!/usr/bin/env python3
"""What is in a saved state, and how much of it is the same bytes twice.

The breakdown save_state prints says where the weight is by mapping.  This asks
the next question: the device arena holds what ONNX Runtime copied there from
its own heap, so if those bytes are still in the heap as well, the arena's half
need not be carried - a resume could copy it across again.  Whether that is
true is a matter of comparing them, and the file already has both.

    python tools/stateanalyse.py ~/vv/session.state
"""
import collections
import hashlib
import struct
import sys

PAGE = 4096
# The shim's arena, from src/cudahost.cpp.  A *range*, not a floor: the guest's
# own heap sits at 0x5555_5555_xxxx, which is the larger number, so "at or above
# the arena" called every page of the heap device memory and reported that none
# of the guest's memory had been saved at all.
ARENA_BASE = 0x0000_3000_0000_0000
ARENA_BYTES = 512 << 20
ARENA_FIRST_PAGE = ARENA_BASE // PAGE
ARENA_LAST_PAGE = (ARENA_BASE + ARENA_BYTES) // PAGE


def is_device(index):
    return ARENA_FIRST_PAGE <= index < ARENA_LAST_PAGE


def sections(data):
    """Yields (tag, payload) for each section of a saved state."""
    if data[:8] != b"X86EMUST":
        raise SystemExit("not a saved state")
    version, _ = struct.unpack_from("<II", data, 8)
    if version != 1:
        raise SystemExit(f"version {version}, this tool reads 1")
    at = 16
    while at + 12 <= len(data):
        tag = data[at:at + 4]
        (length,) = struct.unpack_from("<Q", data, at + 4)
        at += 12
        if tag == b"END ":
            return
        yield tag, data[at:at + length]
        at += length


def main(path):
    data = open(path, "rb").read()
    print(f"{path}: {len(data) / 1048576:.1f} MB")

    pages = {}
    for tag, body in sections(data):
        if tag != b"PAGE":
            continue
        (count,) = struct.unpack_from("<Q", body, 0)
        at = 8
        for _ in range(count):
            (index,) = struct.unpack_from("<Q", body, at)
            pages[index] = body[at + 8:at + 8 + PAGE]
            at += 8 + PAGE
        break
    if not pages:
        raise SystemExit("no pages in this state")

    # Where the pages actually are, rather than where they were assumed to be.
    # Runs of consecutive indices, so the address space reads as the handful of
    # mappings it is rather than forty thousand numbers.
    order = sorted(pages)
    runs = []
    start = prev = order[0]
    for index in order[1:]:
        if index != prev + 1:
            runs.append((start, prev))
            start = index
        prev = index
    runs.append((start, prev))
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    print("  the carried pages, by run:")
    for first, last in runs[:10]:
        n = last - first + 1
        print(f"    {first * PAGE:#016x} .. {(last + 1) * PAGE:#016x}"
              f"  {n:7d} pages  {n * PAGE / 1048576:8.1f} MB")
    if len(runs) > 10:
        rest = sum(last - first + 1 for first, last in runs[10:])
        print(f"    and {len(runs) - 10} shorter runs, {rest * PAGE / 1048576:.1f} MB")

    arena = {i: p for i, p in pages.items() if is_device(i)}
    guest = {i: p for i, p in pages.items() if not is_device(i)}
    print(f"  {len(pages)} pages carried"
          f"  =  {len(guest)} guest ({len(guest) * PAGE / 1048576:.1f} MB)"
          f"  +  {len(arena)} device ({len(arena) * PAGE / 1048576:.1f} MB)")

    # A page of the arena that is byte-for-byte a page the guest also has is one
    # a resume could copy rather than read.  Hashed rather than compared because
    # there are tens of thousands of each.
    guest_hashes = collections.defaultdict(int)
    for p in guest.values():
        guest_hashes[hashlib.blake2b(p, digest_size=16).digest()] += 1

    shared = sum(1 for p in arena.values()
                 if hashlib.blake2b(p, digest_size=16).digest() in guest_hashes)
    print(f"  {shared} of {len(arena)} device pages are also in the guest's memory"
          f"  ({shared * PAGE / 1048576:.1f} MB)")

    # And how much of the whole file is simply the same page repeated.
    everything = collections.defaultdict(int)
    for p in pages.values():
        everything[hashlib.blake2b(p, digest_size=16).digest()] += 1
    distinct = len(everything)
    print(f"  {distinct} distinct pages of {len(pages)}"
          f"  ({distinct * PAGE / 1048576:.1f} MB if each were kept once)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "session.state")
