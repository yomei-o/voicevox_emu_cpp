# Where this is, and what to do next

Working notes. The README says what this *is*; this says what is unfinished and
what is known about it. Written 2026-08-06.

## Start here

    sh setup.sh        # emulator, sysroot, runtime, model, dictionary, guests
    sh run_probe.sh    # ~10 s, should end "PROBE OK"
    sh run_tts.sh      # gets to load_voice_model and fails - that is the open problem

`setup.sh` needs `curl`, `tar`, `ar`, `xz`, and a clang with `ld.lld` beside it
(LLVM 19 here; set `CLANG=`). It needs no Linux host: this was all done on
**ARM64 Windows**, where even the x64 tooling is running under Windows' own
emulation.

## The one open problem

**The encrypted model does not decrypt inside the emulator**, and everything
around it does.

```
step  voicevox_onnxruntime_load_once            ... 3.0 s
step  voicevox_open_jtalk_rc_new                ... 1.0 s   (100 MB sys.dic)
step  voicevox_synthesizer_new                  ... 0.0 s
step  checksum the model file as the guest sees it
      58214379 bytes, fnv1a fbd5370b3d46e74d    ... 14.0 s
step  voicevox_voice_model_file_open            ... 0.0 s
step  voicevox_synthesizer_load_voice_model
FAIL  load_voice_model: 27 モデルデータを読むことができませんでした
      Caused by: Failed to load model because protobuf parsing failed.
```

Four controls have been run, and between them they corner it:

| control | result | what it rules out |
| --- | --- | --- |
| the same `tts.c` built for Windows, run natively | **works** — load 4.0 s, tts 2.4 s, real audio | the model, the runtime, the API sequence |
| the guest's own FNV-1a of `0.vvm` vs the host's | **identical**, `fbd5370b3d46e74d` | the emulated file path |
| a plain `predict_duration.onnx` through the *emulated* ORT | **`CreateSession` succeeds**, inputs `phoneme_list`/`speaker_id`, output `phoneme_length` | ORT's protobuf parser, graph builder, session init |
| syscall trace across the failing load | four `clock_gettime`, nothing else | anything I/O shaped |

So: the bytes arriving are right, ORT is healthy, and the failure is **specific
to the `vv_bin` path and is pure computation**.

### What has already been tried

**AES-NI, implemented and then ruled out.** Disassembling the runtime shows
144 `aesenc`, 16 `aesenclast`, 10 `aeskeygenassist`, 9 `aesimc` — so there is
AES in there, and the emulator had none. The theory was that a crypto library
finding no AES bit in CPUID declines to transform the data rather than falling
back to software, which would produce exactly this error.

Both halves are now in `x86_emu_cpp` and both are verified:

- `src/sse.cpp` implements AESENC/AESENCLAST/AESDEC/AESDECLAST/AESIMC/
  AESKEYGENASSIST and PCLMULQDQ. `tests/aes/aesni.c` checks them against the
  FIPS-197 vectors — encrypt, decrypt round trip, and a carry-less square —
  and prints `AES OK`.
- `src/cpu.cpp` CPUID leaf 1 now sets ECX bits 1 and 25. Verified from inside
  the guest: `AES=1 PCLMUL=1`, with SSSE3/SSE4/AVX still 0 (deliberately —
  glibc's IFUNC reads those and would switch memcpy to code that is not here).

**It changed nothing.** Same failure, byte for byte. So either the decrypt
path never consults the AES bit, or it is never reached.

### Where to look next, roughly in order

1. **Find out whether an AES instruction executes at all.** This is the cheapest
   remaining fact and it splits the problem in half. Add a one-shot
   `fprintf(stderr, ...)` on the first AES opcode in `sse.cpp` (or an env-gated
   counter) and run `run_tts.sh`. If AES never fires, the decrypt is not
   running; if it fires, the decrypt is running and something after it is wrong.
2. **`RDTSC`.** `cpu.cpp` returns `instructions_executed` as the timestamp
   counter. Code that guards a secret sometimes times itself, and an
   interpreter's timings are absurd. Making RDTSC track a real clock is easy
   and would settle it.
3. **`CPUID` beyond leaf 1.** The emulator reports max basic leaf **1** and
   answers zero for everything else — including leaf 7, leaf 0xD (XSAVE) and
   the cache leaves. Real code sometimes treats "leaf missing" differently
   from "feature absent", and `xgetbv` is worth checking for too.
4. **The 57 seconds.** The failing load takes ~57 s emulated (~1 G
   instructions) against 4 s native. That is a lot of work to do before giving
   up, which argues the decrypt *is* running and producing wrong bytes rather
   than being skipped. Worth pinning down with (1).
5. If it comes to it, the honest fallback is the **Windows DLLs**, which are
   known to work natively on this machine. 23 imports are missing rather than
   0, but the failure mode would at least be different.

### Where the line is

Running the runtime is its intended use. Reading a decrypted model back out of
guest memory is not, and the voice-model terms — separate from CORE's MIT
licence — forbid it. Diagnosing *which instructions the emulator must
implement* is emulator bring-up and is fine; stepping into the decryption to
learn what it does is not. Everything above stays on the first side of that.

## Fixed on the way here (both in `x86_emu_cpp`)

**`statx`, syscall 332** — this one is worth remembering. `synthesizer_new`
died reading address zero:

```
[sys] unimplemented syscall 332
[sys] 332(ffffff9c,7fffffefdfb8,100,fff,...) = -38    statx -> ENOSYS
[sys] 262(ffffff9c,7fffffefdfb8,...)         = -2     fstatat fallback
[sys] unimplemented syscall 332
[sys] 332(0,0,0,fff,0,1)                     = -38    the probe, with NULLs
x86emu: unmapped memory read at 0x0000000000000000
```

Rust's standard library decides whether `statx` exists by **calling it with
NULL pointers** and reading errno: ENOSYS or EPERM means "fall back to
`fstat`", anything else means present. A real kernel says EFAULT. Answering
ENOSYS sent glibc down its generic fallback, which called `fstatat` with a
NULL pathname, which the emulator handed to `read_cstring`.

The lesson generalises: **"not implemented" is not a neutral answer.** A caller
that handles ENOSYS takes a *different* path, and that path can be worse than
the one you declined to support.

`syscalls.cpp` maps 332 (and i386's 383); `syscalls_files.inc` fills a real
`struct statx` and both `Statx` and `Newfstatat` now return EFAULT for a null
pathname or buffer. `tests/run_tests.sh` still passes 45/45.

**AES-NI and PCLMULQDQ** — above. They did not fix the problem but they are
correct, tested, and the emulator is better for having them.

## Still unimplemented, seen in the trace but harmless so far

- **syscall 25, `mremap`** — glibc's `realloc` tries it on the 58 MB model
  buffer, gets ENOSYS, and correctly falls back to mmap + copy + munmap (you
  can see all three in the trace). Worth implementing anyway; see the `statx`
  lesson.
- **syscall 99, `sysinfo`** — once, at startup.
- `/etc/localtime` is absent, so the log timestamps come out as UTC. Harmless.

## Speed, when it does start working

Measured here: the emulator retires **~17 M instructions/s** (500 M in 29.9 s),
and CPython's interpreter loop runs **~88× slower** than native. From
`voicevox_core_cpp`'s own figure — HiFi-GAN decode at 2.5 s for 1.05 s of audio
with Eigen AVX2 on four cores — an emulated single-threaded SSE2 run works out
at roughly **half an hour per second of audio**.

Two things bend that, in opposite directions:

- This host is **ARM64 Windows under QEMU**, so `x86emu.exe` is itself being
  emulated. On real x86-64 the 17 MIPS should be several times higher. The 88×
  *ratio* is fair — both sides pay the same tax — but the absolute seconds are
  pessimistic.
- `Memory::host_ptr` does an `unordered_map` lookup on **every guest memory
  access**, with no TLB. A 1 GB working set is a quarter-million pages in a
  hash table. A one-entry last-page cache is the obvious first move and helps
  every guest, not only this one.

Seconds rather than minutes needs a JIT, which is a different project.

## Why this shape

**Linux guests, not Windows.** The Windows surface looked nearly done — of 689
imports across the two DLLs, dropping `MSVCP140`/`VCRUNTIME140` (loaded for
real) leaves 263 symbols with only 23 unhooked. The Linux `.so`s won anyway:
the contract is the syscall ABI rather than an open-ended Win32 surface that
`GetProcAddress` can extend at runtime, there is no VS redistributable to find,
and the Rust core's Windows-only demands (`ProcessPrng`, `oleaut32`, `dxgi`,
`WaitOnAddress`) all evaporate. The clincher was that `x86_emu_cpp` had already
done glibc's `ld.so` — the ISA gate, IFUNC, `MAP_FIXED` semantics — so `probe`
passed with **no emulator changes at all**.

**x86 and not `aarch64_emu_cpp`.** Considered seriously. Against it: its Linux
side is musl-only while ORT's arm64 build is glibc-linked, and its SIMD is
scalar FP plus a subset of vector ops — no `FMLA`/`FMUL` vector forms, which is
precisely what MLAS is made of. That is a much larger hole than the two things
x86 needed. Revisit if NEON gets filled in; the macOS/dyld work there is
genuinely attractive.

**Threads are avoided, not solved.** `cpu_num_threads = 1`, `ORT_SEQUENTIAL`,
`allow_spinning=0` — the emulator schedules guest threads cooperatively, so a
spinning ORT thread pool would never yield. No `clone` has appeared in any
trace, so the assumption holds. If one ever does, glibc's NPTL on the dynamic
path is untested (x86_emu_cpp's own notes say so) and would land on the
critical path.

## Things that cost time, so they are written down

- **MSYS rewrites guest paths.** `-Wl,-rpath,/opt/vv` became
  `RUNPATH C:/Program Files/Git/opt/vv`, and `/opt/vv/probe` on the command
  line became a host path. `MSYS2_ARG_CONV_EXCL='*'` and `MSYS_NO_PATHCONV=1`
  on both the compiler and the emulator; the run scripts set them.
- **The program argument is a host path, the guest's arguments are not.**
  `x86emu --sysroot sysroot sysroot/opt/vv/tts /opt/vv/0.vvm`.
- **Debian does not ship the SONAME symlinks** — ldconfig makes them at install
  time — and Windows cannot create the symlinks that *are* in the archive.
  `make_sysroot.sh` materialises both as copies. A dangling one presents as
  "cannot open shared object file: Invalid argument" on a file that is there.
- **Redirecting the emulator's stdout hides progress.** The guest's `fflush`
  flushes the guest; the emulator's own stdout is block-buffered into the file.
  Watch long runs on a terminal.
- Non-ASCII on the command line is not worth the encoding risk; `tts.c` has the
  default text compiled in as UTF-8.

## The vendored emulator

`x86_emu_cpp/` here is a copy of the sibling project, carrying the `statx` and
AES-NI changes. `setup.sh` prefers a built `../x86_emu_cpp/x86emu.exe` if there
is one, so on the machine that has both, the sibling checkout is the source of
truth and this copy is for anyone who clones only this repository. **Changes to
the emulator should go upstream too** — they are not voicevox-specific.
