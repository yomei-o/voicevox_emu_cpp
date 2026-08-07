# voicevox_emu_cpp

The official VOICEVOX CORE, running inside an emulator — and the same C API on
the outside.

VOICEVOX's released voice models are encrypted. They are `vv_bin` payloads that
only [`voicevox_onnxruntime`](https://github.com/VOICEVOX/onnxruntime-builder) —
a patched ONNX Runtime whose source repository is private — can open. That is
why [voicevox_core_cpp](../voicevox_core_cpp), a from-scratch reimplementation
with its own ONNX engine, can run the whole pipeline and still not say a word:
there is no plaintext model for it to run.

This project takes the other road. Rather than reimplement the runtime, it
**runs the official binaries as they are**, one x86-64 instruction at a time,
under [x86_emu_cpp](../x86_emu_cpp). Nothing is decompiled and nothing is
extracted: the library does its own work, in its own address space, exactly as
its authors intended — only the CPU underneath it is software.

Which makes the interesting property portability rather than speed. The guest is
a Linux x86-64 program; the host is whatever the emulator builds for, which today
includes Windows on ARM and a browser tab.

**Demo:** <https://yomei-o.github.io/voicevox_emu_cpp/web/> — type Japanese and
watch the real Open JTalk work out the accent inside an emulated CPU, in about a
second. The synthesis button is there too, and honest about costing hours.

**And a second one**, [`web/cuda.html`](web/cuda.html), where synthesis takes
**twenty-two seconds** instead: it runs the *CUDA* build of the runtime and
answers its kernels with compiled WebAssembly rather than interpreting them.
It needs the CUDA libraries, which this repository does not carry, so the page
asks for them from disk - nothing leaves the tab. Building the sessions still
takes about thirteen minutes and is now the whole cost of a run.

## State

**It speaks.** `voicevox_synthesizer_load_voice_model` decrypts the model and
`voicevox_synthesizer_tts` returns a WAV, through the official binaries, on a
CPU that is entirely software.

**The API is the same one.** `voicevox_core.dll` here exports VOICEVOX CORE's C
API — all 63 functions of the published `voicevox_core.h`, with the same
signatures and the same meaning. A program written against CORE links this
instead and runs unchanged. `src/apitest.c` calls the whole surface and prints
what comes back, and it has been run three ways:

| | |
| --- | --- |
| against the real `libvoicevox_core.so` | 45 checks, 0 failed — the reference |
| against this implementation, natively | the same 45, and the same values; the two outputs differ only in a randomly generated UUID |
| against this implementation, **through the emulator** | `--no-audio`: 42 checks, 0 failed, every line identical to the reference. The three it skips are the ones that generate audio, which take hours emulated and which `vvsay` covers end to end. |

**It runs in a browser.** `web/` builds the emulator to WebAssembly and runs the
same guest in a tab: the real `ld-linux-x86-64.so.2` maps the 18 MB runtime,
IFUNC dispatch settles, the C++ static initialisers run, ONNX Runtime comes up.
Text analysis - Open JTalk working out the moras and the accent - answers in
about a second there, which makes it the half of VOICEVOX a browser can do
interactively. Synthesis in a tab is real but takes hours, so the page ships the
WAVs this machine produced and says exactly what they cost.

    node web/test_page.mjs          # the page's own path, in node
    node tools/browser_test.mjs     # the page, in a headless browser
    node tools/browser_test.mjs --url https://yomei-o.github.io/voicevox_emu_cpp/web/

Run the instruction check after touching the emulator, both ways — a
WebAssembly build is a different compiler on a different target, and undefined
behaviour that happens to be right on x86 is not right there, which is exactly
how the conversion bug below was found:

    sh tools/check_isa.sh          # the native build
    sh tools/check_isa.sh wasm     # and the browser one

### What it took

The model would not decrypt, and four controls had cornered it: the bytes
arriving were right, ORT was healthy, no I/O was involved, and AES-NI — added on
suspicion — turned out never to execute. What was left was that some instruction
computed the wrong answer.

`src/isatest.c` found them. It runs 240 groups of instructions over operands
chosen to hit the edges, folds every result and every architecturally *defined*
flag into a checksum, and prints one line per group. Run natively, under
`qemu-x86_64` and under `x86emu`, the first two agree exactly — which is what
makes a third answer mean something. Five bugs came out:

| | |
| --- | --- |
| **PSLLW/PSLLD/PSLLQ ↔ PSRAW/PSRAD** | `0F 71..73`'s `/reg` field is `/2` PSRL, `/4` PSRA, `/6` PSLL. Four and six were swapped, so **every PSLLQ was an arithmetic shift right**. Compilers almost never emit these; hand-written SIMD is made of them. This was the one. |
| **MINPS/MAXPS/MINPD/MAXPD** | The definition is `SRC1 < SRC2 ? SRC1 : SRC2`, and the `else` carries both the NaN case and the tie — which is how MINPD tells `+0.0` from `-0.0`. Testing `b < a` gets both wrong. |
| **SHLD/SHRD with a count of zero** | Nothing shifts and no flag changes, but the destination register is still written, and a 32-bit write zeroes the upper half. Returning early left the old upper half in place. |
| **SHLD/SHRD's OF** | Defined for a count of one, and unimplemented. |
| **CVTPS2DQ and friends, in the wasm build only** | The conversion was a plain `static_cast<int32_t>(double)`, undefined when the value does not fit or is a NaN. Compiled for x86 that cast *becomes the instruction being emulated*; compiled to WebAssembly it becomes a trapping or saturating truncate. `to_int32_x86`/`to_int64_x86` in `cpu.h` now say what x86 means, and x87's FIST used the same pattern. |

Two more emulator gaps turned up on the way: `statx` (syscall 332), and guest
writes to stdout not being flushed out of the emulator's own stdio buffer, which
deadlocked anything talking to the guest through a pipe.

## Getting there

    sh setup.sh          # unpack, build the emulator, build the guests
    sh run_probe.sh      # does the runtime come up?            (~10 s)
    sh run_tts.sh        # text in, sysroot/opt/vv/out.wav out

**A clone is self-contained.** The runtime, the core, ずんだもん's voice model,
the Open JTalk dictionary and a Debian sysroot are all here; `setup.sh` downloads
nothing. See `licenses/README.md` for what is redistributed under what terms.

No Linux host and no WSL is required: the sysroot is unpacked Debian amd64
packages, so a Windows machine builds and runs an x86-64 Linux guest. (Where WSL
*is* available its gcc builds the guests directly, which is simpler than the
clang cross-compile, and `qemu-x86_64` there makes an excellent reference.)

## Synthesis in seconds

Needs the CUDA build of `voicevox_onnxruntime` (which this repository does not
carry) and a Linux host or WSL. See [CUDA_SHIM.md](CUDA_SHIM.md) for what it is
and what it measures.

    sh tools/slim_provider.sh <providers_cuda.so>   # 440 MB -> 9.3 MB gzipped
    MODE=guest sh tools/make_cuda_stubs.sh <lib dir>
    sh build_cudaemu.sh                             # the emulator + the shim
    sh tools/wslrun_cuda.sh                         # and a WAV, in seconds

`MODE=native` on the same generator builds a shim that does the arithmetic in
the calling process instead, which is the one to debug against: it runs in a
second, and `sh tools/check_shim.sh` holds it to sixteen reference values that
a Tesla T4 and a desktop CPU both produce.

## The drop-in API

    sh build_api.sh      # guest/vvagent and voicevox_core.dll

```c
#include "voicevox_core.h"          // the official header, unmodified

const VoicevoxOnnxruntime *ort;
voicevox_onnxruntime_load_once(opts, &ort);
voicevox_synthesizer_tts(syn, "ずんだもんなのだ", 3, topts, &len, &wav);
```

Link `voicevox_core.dll` instead of the real one and that code works on a machine
where no build of CORE exists.

Underneath, each call is packed into a frame and answered by a guest agent
running inside the emulator with the official `libvoicevox_core.so` linked in:

```
your program → voicevox_core.dll → x86emu → vvagent → libvoicevox_core.so
```

Handles are guest pointers passed through untouched; JSON and WAV buffers are
copied into the host heap and freed by the host's own `voicevox_json_free` /
`voicevox_wav_free`. Paths that lie inside `sysroot/` are rewritten to their
guest form, and paths that already look guest-absolute are left alone.

`nativeemu.sh` stands in for the emulator on a Linux x86-64 machine, which turns
a round trip from minutes into milliseconds — the only practical way to debug 63
entry points.

`run_vvsay.sh` runs the whole thing end to end and sets the environment the host
library needs, including the two MSYS variables without which a guest path
arrives as `C:/Program Files/Git/opt/vv/...`:

    sh build_api.sh
    sh run_vvsay.sh text_short.txt zundamon.wav

## Speed, honestly

The emulator is an interpreter, and this is what that costs:

| | native | `qemu-x86_64` | `x86emu` | `x86emu` in wasm |
| --- | --- | --- | --- | --- |
| runtime comes up (`probe`) | — | — | 4 s | 13 s |
| dictionary load + text analysis | 0.0 s | — | 1.0 s | 1.0 s |
| model decrypt + session init | 0.6 s | 6.4 s | 8-9 min | 23 min |
| `tts`, 0.52 s of audio | 0.5 s | — | 42 min | 99 min |
| `tts`, 1.45 s of audio | 1.5 s | 517 s | 118 min | — |

The whole of "あ" in a browser is **44.4 G instructions** — 85 to 123 minutes
depending on what else the machine is doing, at some six million instructions a
second against the billions a real CPU manages. `tools/browser_test.mjs --speak`
drives the demo page in a headless browser, waits it out, and pulls the WAV back
out of the page; the bytes it gets are **identical** to the x86-64 build's, and
within 3 of 3912 of native `libvoicevox_core`. Slow, and the same answer.

And the audio that comes out is the real thing. The same utterance was made five
ways - on a Tesla T4 through the CUDA build, on a Linux CPU, under `x86emu`, in a
browser tab, and through the CUDA build on a machine with **no GPU at all** - and
`tools/wavcmp.mjs` puts every pair within **8 of 12988**, under a tenth of a
percent. Not "it ran under an emulator": the same answer as the dedicated
hardware. What difference there is comes from `RSQRTPS` and `RCPPS` being
*approximate* instructions that hardware answers to twelve bits and this emulator
computes exactly.

That fifth way is [CUDA_SHIM.md](CUDA_SHIM.md): run the *CUDA* build of
voicevox_onnxruntime against stand-in libcudart / cuBLAS / cuDNN and do the
arithmetic natively with Eigen. ONNX Runtime registers 4757 kernels; an utterance
launches 43, in 20 loop shapes, plus eight library functions that compute.
Filling those in gets the T4's own output to within 2 samples of 12988.

**And that is what makes synthesis fast under the emulator.** The same binary,
run twice - once with the arithmetic interpreted like everything else, once with
it handed to the host through one reserved syscall:

| "あ", 0.52 s of audio | synthesis | against the T4 |
| --- | --- | --- |
| everything interpreted | 3476 s (58 minutes) | 3 of 3912 |
| the arithmetic on the host | **5 s** | 3 of 3912 |

Seven hundred times, for the same answer. It costs nothing to move the tensors
because they never move: a CUDA device pointer is not host memory and nothing in
ONNX Runtime may dereference one, so device memory can simply *be* host memory.
Over a whole utterance the boundary crossings - 1983 of them - come to 0.06 s.

Two things make the gap that large. The interpreter retires tens of millions of
instructions a second where the machine does billions; and CPUID here advertises
SSE2 only, so ORT's kernels take their slowest path — which is also true of qemu,
and is why qemu's own numbers are nothing like native.

The second of those looked like the lever and is not. SSSE3 and SSE4.1 are
implemented now and verified (isatest, 303 groups, native and wasm both matching
a real CPU and qemu), so advertising them is two lines. With them advertised the
whole pipeline still works — and the WAV that comes out is *bit-identical* to the
SSE2-only one, which says ONNX Runtime did not change which kernels it runs. The
fast paths want AVX2 and FMA, which is a much larger job, though a bounded one
now that there is a way to check each instruction.

A one-entry-per-slot page cache in `Memory::host_ptr` (this repo's copy of the
emulator has it) took the worst of the hash-table lookups out, worth about 25 %
on a memory-bound guest. Seconds rather than minutes needs a JIT, which is a
different project.

## Layout

| | |
| --- | --- |
| `src/probe.c` | does the runtime load and initialise at all? |
| `src/tts.c` | the whole thing: text in, `out.wav` out, timed per step |
| `src/vvrpc.h`, `src/vvagent.c`, `src/vvhost.c` | the drop-in API: wire format, guest half, host half |
| `src/apitest.c` | calls the whole API surface and prints what comes back |
| `src/vvsay.c` | the smallest program that uses the API |
| `src/isatest.c` | the instruction differential test that found the bug |
| `src/memtest.c` | large-buffer allocator test (ruled the allocator out) |
| `web/` | the emulator as WebAssembly, and the browser demo |
| `setup.sh`, `build.sh`, `build_api.sh` | build the guests, the emulator, the API |
| `run_probe.sh`, `run_tts.sh` | run them under the emulator |
| `sysroot/`, `guest/` | the committed payload |
| `make_sysroot.sh`, `fetch_models.sh` | rebuild the payload. Not needed for a clone. |
| `x86_emu_cpp/` | a copy of the emulator, carrying the fixes above |

## Requirements

- a C++17 compiler for the emulator. `setup.sh` uses `g++` when there is one and
  CMake with MSVC when there is not, which on Windows is the ordinary case.
  `vcvars` is never involved — on the machine this was written on it hangs.
- to build the guests: WSL's gcc, or clang with `ld.lld` (set `CLANG=`)
- to build the browser demo: emscripten (set `EMCC=`)
- to check the emulator against a reference: a Linux x86-64 machine with
  `qemu-x86_64`, which WSL provides

Every script in the repository has been run end to end on a Windows machine with
MSVC, WSL and emscripten — which is how three of them turned out not to work.

## Terms

`licenses/README.md` has the full table. The short version: everything committed
here permits redistribution, and audio produced with a VOICEVOX voice library
must be credited — `VOICEVOX:ずんだもん` and the like. See
<https://zunko.jp/con_ongen_kiyaku.html>.

What the voice-model terms *do* forbid is 「逆コンパイル・リバースエンジニアリング
及びこれらの方法の公開すること」. Running the runtime is its intended use and is not
that. Reading a decrypted model back out of guest memory would be, and this
project does not do it — the emulator's `--dump` is pointed at nothing here for a
reason. Finding out *which instruction an emulator computes wrongly* is emulator
bring-up, and `isatest.c` does it without looking at the model at all.
