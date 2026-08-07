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
- **Synthesis in seconds, through the CUDA build.** `tools/wslrun_cuda.sh` runs
  the CUDA runtime inside `vvcudaemu` - the emulator with the shim behind one
  reserved syscall - on a machine with no GPU: **5 s** for "あ" against 3476 s
  with the same arithmetic interpreted, and the audio is within 3 of 3912 of a
  Tesla T4's own. All 377 kernel launches an utterance makes are handled. The
  tensors never move: a device pointer is host memory, so the 1983 boundary
  crossings in a run come to 0.06 s between them. [CUDA_SHIM.md](CUDA_SHIM.md).
- **The API is complete.** All 63 functions of `voicevox_core.h`. `apitest.c`
  built against the real library and against this one prints the same 45 checks
  and the same values, differing only in a random UUID.
- **The browser build runs, all the way to audio.** `tools/browser_test.mjs
  --speak` drives the demo page in a headless browser, waits out the 85 minutes,
  pulls the WAV back out of the page and checks it: **identical** to what the
  x86-64 build of the emulator produces, and within 3 of 3912 of native
  `libvoicevox_core`. Text analysis there answers in under two seconds.
  `web/test_page.mjs` rehearses the same path in node without a browser.

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

`git diff 8e96b56 HEAD -- x86_emu_cpp/` is the whole of it: 8 files, +572/-61.

| file | what |
| --- | --- |
| `src/sse.cpp` | **the PSLL/PSRA fix**; MIN/MAX ties and NaN; `RSQRTPS`/`RSQRTSS`/`RCPPS`/`RCPSS`; the x86 float-to-int conversions; **all of SSSE3 and SSE4.1** (plus `PCMPGTQ`); AES-NI and its `X86EMU_AES_COUNT` counter |
| `src/cpu.h` | `to_int16_x86` / `to_int32_x86` / `to_int64_x86` |
| `src/cpu.cpp` | `SHLD/SHRD` zero-count write and OF; CPUID leaf 1 ECX bits 1 and 25; the prefix decoder as a switch; `census()` resolved once |
| `src/x87.cpp` | FIST/FISTP through the conversion helpers |
| `src/memory.{h,cpp}` | the direct-mapped page cache in `host_ptr` |
| `src/emulator.cpp` | flush the host's stdout after a guest write |
| `src/files.cpp` | the same for a child's redirected stdout |
| `web/wasm_api.cpp` | `emu_set_sysroot` (this repo's `web/` copy) |

And a second round, for the CUDA shim - also all general, none of it
voicevox-specific:

| file | what |
| --- | --- |
| `src/memory.{h,cpp}` | pages are **reserved, not allocated**; a whole page of zeros written over an untouched page is skipped. Peak for the CUDA run: 1058 MB -> 538 MB, measured both ways with `tools/wslmemab.sh` |
| `src/syscalls.cpp` | a file mapping is read a megabyte at a time, not into one buffer the size of the segment (430 MB here); and the region is **named after the file** |
| `src/emulator.{h,cpp}` | `on_host_call` and one reserved syscall, so an embedder can put host services behind it; `alloc_pages` takes a region name |
| `src/cpu.cpp` | an unsupported opcode says **which mapping** it is in |
| `src/files.{h,cpp}` | `path_of(fd)`, which is where the mapping's name comes from |

The naming paid for itself on its first run: the AVX instruction that stopped
everything was not in ONNX Runtime, it was in *our own* libcudart stand-in.

`statx` (syscall 332) landed in `syscalls.cpp` / `syscalls_files.inc` before this
project's second session and is already in the vendored copy.

Regression checks that must stay green: the sibling checkout's
`tests/run_tests.sh` (7/7 here), and `sh tools/check_isa.sh` plus
`sh tools/check_isa.sh wasm` (303/303 each), and `sh tools/regress_tts.sh`
- the whole CPU-path pipeline, about seventy minutes, ending bit-identical
to `web/sample/emu_zundamon.wav`. That last one deletes its output first,
deliberately: a run that is cut short otherwise leaves the previous one in
place, and comparing *that* against the reference it was copied from
reports zero difference and means nothing. It did exactly that once. `src/isatest.c`, `src/memtest.c`
and `tools/check_isa.sh` belong upstream too - they are emulator tests, not
voicevox ones.

**Merge policy.** Work on the emulator here, merge back upstream in one
deliberate step. The run scripts now prefer the **vendored** build and print
which one they chose; they used to prefer a sibling checkout, which on this
machine is a build from before all of the above and fails in exactly the way
this project spent a day explaining. `EMU=` still overrides.

## What is unfinished

0. **Read [CUDA_SHIM.md](CUDA_SHIM.md) first.** The speed story below is still
   true of the CPU path, and it is no longer the only path. Running the *CUDA*
   build with the arithmetic handed to the host takes synthesis from **3476
   seconds to 5** for the same utterance, with the same audio. What is left
   there is not arithmetic at all: building the sessions takes 6 m 37 s, all of
   it interpreted, and that is now the whole cost of a run.

   `X86EMU_PROFILE=100000` says where those instructions go, and it closes off
   one idea before anyone builds it. A complete run is 7.8 G instructions:
   **83 % ONNX Runtime, 12 % libvoicevox_core, 3.5 % libc, 1.2 % libstdc++.**
   For an ELF guest the real glibc runs interpreted - the library hooks below
   are only reached through PE imports - so hooking `memcpy` and `malloc`
   natively looked worth a lot. It is worth at most three and a half per cent.

1. **Speed.** This is the whole story for the CPU path. The interpreter retires
   tens of millions of instructions a second; ORT wants billions.

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
2. **CPUID says SSE2 only.** SSSE3 and SSE4.1 are now *implemented* and checked
   (isatest, 303 groups, native and wasm both matching a real CPU and qemu), so
   advertising them is a two-line change rather than a project. It was tried and
   the bits are still off, because measuring said it buys nothing:

   - with `ECX_SSSE3 | ECX_SSE41` set, `probe`, text analysis and a full
     synthesis all work;
   - the WAV that comes out is **bit-identical** to the one the SSE2-only build
     produces. Not merely close - identical. Which says ONNX Runtime did not
     change the kernels it runs for this workload;
   - the timings moved by ±10-15 % in both directions with two other jobs on the
     machine, i.e. noise.

   So the SSE4.1 bit is not the lever. MLAS's fast paths want **AVX2 and FMA**,
   which is a much larger implementation job - but a bounded one now, and
   `isatest.c` is the tool that makes it safe. The old reason for keeping the
   bits off still holds too: glibc's IFUNC reads them and picks a different
   `memcpy`, so turning one on is never only about the library you meant.
3. *(closed)* The browser demo's synthesis path is verified end to end - see
   above. Text analysis is verified in a real browser against both a local
   server and the deployed Pages site (`node tools/browser_test.mjs [--url ...]`).
7. **The guest needs `/tmp`.** Absent, `voicevox_open_jtalk_rc_use_user_dict`
   fails with `USE_USER_DICT_ERROR` - Open JTalk compiles a user dictionary
   through a temporary file. `sysroot/tmp/.keep` carries the directory, and
   `make_sysroot.sh` and the browser worker both create it. Found by running
   `apitest --no-audio` through the emulator, which is what that flag is for.
4. **`voicevox_onnxruntime_init_once`** is stubbed: this build of CORE loads
   ONNX Runtime rather than linking it, so the header does not declare it.
5. **The browser pays the model load again for every utterance**, because
   `emu_run_path` runs a guest to completion and the page starts a fresh one
   each time. Twenty-three minutes, every click. The interesting thing this
   blocks is not audio - the vocoder is hours either way - but
   `voicevox_synthesizer_create_audio_query`, which runs only the two small
   models and would answer in seconds once the sessions exist. Text in, real
   phoneme durations and pitches out, interactively, from the real models.

   What it needs: a persistent `Emulator` behind the wasm API rather than one
   per call, and fd 0 backed by a buffer JS can push into. The guest half
   already exists - `vvagent` speaks exactly this protocol over stdin/stdout,
   and the host library drives it through a pipe today.

   **The one obstacle, read out of the code rather than guessed at.** The
   blocking path is already right: `FileTable::read` answers `kEAGAINPipe` for
   an empty pipe that still has a writer, `Sys::Read` calls
   `Emulator::block_syscall_retry`, and that steps `rip` back two bytes (both
   `syscall` and `int 0x80` are two bytes), marks the thread `Blocked` with a
   predicate, and yields. So far so good. What happens next is the problem: the
   guest agent is a single thread, so nothing else is runnable, and
   `processes.cpp` declares *"all processes are blocked: deadlock"* and throws.
   Which is the correct answer for every guest that exists today - the host is
   not part of the picture.

   So the change is: a way to tell the scheduler that a particular blocked
   thread is waiting on the *embedder*, and that this is a reason to return
   control rather than to declare deadlock. Small in lines, delicate in effect -
   the deadlock detector is there to catch real bugs, and weakening it blindly
   would hide them. Worth doing deliberately, with the guest agent's own
   round trip (`VV_OP_PING`, then `analyze`, both about a second) as the test
   before anything touches the model.
6. **Threads are avoided, not solved.** `cpu_num_threads = 1`,
   `ORT_SEQUENTIAL`, `allow_spinning=0`. No `clone` has appeared in any trace.

## The Pages site fell five hours behind, and what got it moving

After a stretch of rapid pushes the site stopped publishing: every file it
served carried the same `Last-Modified`, and five pushes over five hours
produced no new deployment, while the repository moved on.

An **empty commit did not help** - which makes sense, since its tree is
identical to its parent's and there is nothing new to publish. A push that
actually changed a file did, and it published everything at once.

So the working rule is: **if Pages looks stuck, push a real change, not an empty
commit.** Whether the underlying cause was the ten-builds-an-hour limit expiring
at about the same moment, or the unchanged tree being skipped, is not something
this could distinguish from outside - the Actions tab would say. Either way,
batching pushes is the thing that avoids it.

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
