# Where this is, and what to do next

Working notes. The README says what this *is*; this says what is known about it
and what is unfinished. Rewritten 2026-08-06 after the open problem closed.

## Start here

    sh setup.sh          # unpack, build the emulator, build the guests
    sh run_probe.sh      # ~10 s, ends "PROBE OK"
    sh build_api.sh      # guest/vvagent and voicevox_core.dll
    node web/test_page.mjs   # the browser path, text analysis, ~1 s

A clone is self-contained: the runtime, the core, the voice model, the
dictionary and a Debian sysroot are all committed and `setup.sh` downloads
nothing.

## What works

- **The model decrypts and the pipeline produces audio.** `vvsay.exe` →
  `voicevox_core.dll` → `x86emu` → the official `libvoicevox_core.so`, a 25132
  byte WAV. Model load 498 s, inference 2498 s for 0.52 s of audio.
- **The audio is the real thing.** Against a native run of the same text the
  largest sample difference is 3 against a peak of 3912 (0.077 %), rms 0.0078 %.
  `tools/wavcmp.mjs` does the comparison. It is not bit-identical and cannot be:
  `RSQRTPS`/`RCPPS` are *approximate* by definition, hardware answers them to
  about twelve bits and this emulator computes them exactly.
- **The API is complete.** All 63 functions of `voicevox_core.h`. `apitest.c`
  built against the real library and against this one prints the same 45 checks
  and the same values, differing only in a random UUID.
- **The browser build runs.** `probe` and text analysis both work under
  WebAssembly; `web/test_page.mjs` rehearses the page's exact path in node.

## How the decrypt was found, and why it matters for next time

Four controls had already ruled out the model, the runtime, the API sequence,
the file path, ORT's parser and anything I/O shaped. AES-NI was implemented on
suspicion and a counter later showed it never executes. `memtest.c` ruled out
the allocator. What was left was "some instruction computes the wrong answer",
and the way to find *which* was not to diff a multi-gigabyte trace.

`src/isatest.c` runs 240 groups of instructions over operands chosen to hit the
edges and folds every result and every architecturally **defined** flag into a
checksum. The mask per group is the point: a real CPU and qemu genuinely
disagree about the SF of an `IMUL` or the OF of a multi-bit shift, and comparing
an undefined bit compares nothing. With the masks right, native and
`qemu-x86_64` agree exactly - and only then does a third answer mean something.

It found five bugs, one of which was the cause and one of which only exists in
the WebAssembly build:

- **`PSLLW/PSLLD/PSLLQ` ↔ `PSRAW/PSRAD`.** `0F 71..73`'s `/reg` is `/2` PSRL,
  `/4` PSRA, `/6` PSLL; four and six were swapped, so every `PSLLQ` was an
  arithmetic shift right. Compilers barely emit these. Hand-written SIMD - a
  cipher, a hash - is made of them.
- **`MINPS/MAXPS/MINPD/MAXPD`.** `SRC1 < SRC2 ? SRC1 : SRC2`, exactly; the
  `else` carries the NaN case *and* the tie, which is how `MINPD` tells `+0.0`
  from `-0.0`.
- **`SHLD/SHRD` with a count of zero.** No shift and no flag change, but the
  destination register is still written, and a 32-bit write zeroes the upper
  half.
- **`SHLD/SHRD`'s OF**, defined for a count of one and missing.
- **`CVTPS2DQ`/`CVTTPS2DQ`/`CVTPD2DQ`/`CVTTPD2DQ` in the wasm build only.** The
  conversion was a plain `static_cast<int32_t>(double)`, whose result is
  undefined when the value does not fit or is a NaN. Compiled for x86 that cast
  *becomes the instruction being emulated* and looks perfectly right; compiled
  to WebAssembly it becomes a trapping or saturating truncate and quietly is
  not. `to_int32_x86`/`to_int64_x86` now say what x86 means. Running isatest
  under the wasm build is therefore a check worth keeping:
  `node web/test_node.mjs isatest`.

**The lesson worth keeping: build the oracle before hunting.** Two independent
implementations that agree turn a search into a lookup.

## The emulator changes made here

All of it belongs upstream in `x86_emu_cpp`; none of it is voicevox-specific.

| file | what |
| --- | --- |
| `src/sse.cpp` | the PSLL/PSRA fix; MIN/MAX; `RSQRTPS`/`RSQRTSS`/`RCPPS`/`RCPSS`; AES-NI and its `X86EMU_AES_COUNT` counter |
| `src/cpu.cpp` | `SHLD/SHRD` zero-count write and OF; CPUID leaf 1 ECX bits 1 and 25 |
| `src/memory.{h,cpp}` | the direct-mapped page cache in `host_ptr` |
| `src/emulator.cpp` | flush the host's stdout after a guest write |
| `src/files.cpp` | the same for a child's redirected stdout; `statx` support lives in `syscalls*` |
| `web/wasm_api.cpp` | `emu_set_sysroot` |

Regression checks that must stay green: the sibling checkout's
`tests/run_tests.sh` (7/7 here), and `isatest` against `qemu_ref.txt` (240/240).

**Merge policy.** Work on the emulator here, merge back upstream in one
deliberate step. `setup.sh` prefers a built `../x86_emu_cpp/x86emu.exe` when a
sibling checkout exists, so **delete the sibling binary or set `EMU=`** while
working here, or there is no telling which one ran. The sibling on this machine
does *not* have these changes.

## What is unfinished

1. **Speed.** This is the whole story now. The interpreter retires tens of
   millions of instructions a second; ORT wants billions.

   What helped: the direct-mapped page cache in `Memory::host_ptr`, worth about
   25 % on a memory-bound guest (memtest 8m03s -> 6m23s).

   What did **not**, measured against a build of the previous commit, runs
   interleaved: replacing the prefix decoder's ten-deep `if` chain with a switch
   and hoisting `census()` out of a function-local static. Three interleaved
   rounds of `probe`, which is about four seconds of real interpretation:
   5927/5969/5817 ms before, 5949/6033/5848 ms after. Nothing, or slightly
   worse. Both changes are kept because they are clearer, not because they are
   faster - and the useful part is the negative result: **the per-instruction
   cost is not in the prefix chain or the static guard**, so the next person
   should look at the decode/execute switch and the operand helpers, or accept
   that the answer is a JIT.

   Untried and still plausible: an instruction-fetch cursor so decoding does not
   re-resolve the code page for every byte.
2. **CPUID says SSE2 only**, so ORT takes its slowest kernels. Advertising
   SSSE3/SSE4/AVX2 would cut the instruction count a great deal, but every one
   of those instructions then has to be right - and `isatest.c` is exactly the
   tool for that now. Note the reason the bits are off today: glibc's IFUNC
   reads them and switches `memcpy` to code that is not implemented.
3. **The browser demo's synthesis path is unverified end to end** at the time of
   writing - `web/test_page.mjs speak` was still running, and started before the
   conversion fix above, so its output may not be trustworthy. Text analysis is
   verified in a real browser, against both a local server and the deployed
   Pages site (`node tools/browser_test.mjs [--url ...]`).
6. **The guest needs `/tmp`.** Absent, `voicevox_open_jtalk_rc_use_user_dict`
   fails with `USE_USER_DICT_ERROR` - Open JTalk compiles a user dictionary
   through a temporary file. `sysroot/tmp/.keep` carries the directory, and
   `make_sysroot.sh` and the browser worker both create it. Found by running
   `apitest --no-audio` through the emulator, which is what that flag is for.
4. **`voicevox_onnxruntime_init_once`** is stubbed: this build of CORE loads
   ONNX Runtime rather than linking it, so the header does not declare it.
5. **Threads are avoided, not solved.** `cpu_num_threads = 1`,
   `ORT_SEQUENTIAL`, `allow_spinning=0`. No `clone` has appeared in any trace.

## Things that cost time, so they are written down

- **MSYS rewrites guest paths.** `/opt/vv/x` on a command line becomes
  `C:/Program Files/Git/opt/vv/x` - which is what "cannot open shared object
  file" meant the first time the API was run from git-bash. Set
  `MSYS2_ARG_CONV_EXCL='*'` and `MSYS_NO_PATHCONV=1` for the compiler, the
  emulator *and* any host program taking guest paths.
- **The emulator's own stdout was block-buffered.** A guest that answers on
  stdout and a host that waits for the answer deadlocked through a pipe, and a
  long run's progress was invisible through a redirect. Both were the same
  missing `fflush`, now in `Emulator::host_write`.
- **Non-ASCII on a command line is not worth the encoding risk.** `vvsay` reads
  its text from a file for this reason; in the browser argv is exact and the
  text goes straight through.
- **Debian does not ship the SONAME symlinks** - ldconfig makes them at install
  time - and Windows cannot create the ones that *are* in the archive.
  `make_sysroot.sh` materialises both as copies. A dangling one presents as
  "cannot open shared object file: Invalid argument" on a file that is there.
- **`*.txt` is `eol=lf` in `.gitattributes`** because `qemu_ref.txt` is diffed
  against output the emulator writes with LF.
- **`nativeemu.sh`** stands in for the emulator on Linux x86-64 and turns an API
  round trip from minutes into milliseconds. Debugging 63 entry points any other
  way is not practical.

## Where the line is

Running the runtime is its intended use. Reading a decrypted model back out of
guest memory is not, and the voice-model terms - separate from CORE's MIT
licence - forbid it. Diagnosing *which instruction an emulator computes wrongly*
is emulator bring-up and is fine; `isatest.c` did it without looking at the
model at all. The emulator's `--dump` is pointed at nothing here for a reason.
