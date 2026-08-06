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

## State

**It speaks.** `voicevox_synthesizer_load_voice_model` decrypts the model and
`voicevox_synthesizer_tts` returns a WAV, through the official binaries, on a
CPU that is entirely software.

**The API is the same one.** `voicevox_core.dll` here exports VOICEVOX CORE's C
API — all 63 functions of the published `voicevox_core.h`, with the same
signatures and the same meaning. A program written against CORE links this
instead and runs unchanged. `src/apitest.c` calls the whole surface and prints
what comes back; built against the real library and against this one, the two
outputs differ only in a randomly generated UUID.

**It runs in a browser.** `web/` builds the emulator to WebAssembly and runs the
same guest in a tab: the real `ld-linux-x86-64.so.2` maps the 18 MB runtime,
IFUNC dispatch settles, the C++ static initialisers run, ONNX Runtime comes up.
Text analysis - Open JTalk working out the moras and the accent - answers in
about a second there, which makes it the half of VOICEVOX a browser can do
interactively. Synthesis in a tab is real but takes hours, so the page ships the
WAVs this machine produced and says exactly what they cost.

    node web/test_page.mjs        # the page's own path, in node
    node tools/browser_test.mjs   # the page, in a headless browser

### What it took

The model would not decrypt, and four controls had cornered it: the bytes
arriving were right, ORT was healthy, no I/O was involved, and AES-NI — added on
suspicion — turned out never to execute. What was left was that some instruction
computed the wrong answer.

`src/isatest.c` found it. It runs 240 groups of instructions over operands
chosen to hit the edges, folds every result and every architecturally *defined*
flag into a checksum, and prints one line per group. Run natively, under
`qemu-x86_64` and under `x86emu`, the first two agree exactly — which is what
makes a third answer mean something. Four bugs came out:

| | |
| --- | --- |
| **PSLLW/PSLLD/PSLLQ ↔ PSRAW/PSRAD** | `0F 71..73`'s `/reg` field is `/2` PSRL, `/4` PSRA, `/6` PSLL. Four and six were swapped, so **every PSLLQ was an arithmetic shift right**. Compilers almost never emit these; hand-written SIMD is made of them. This was the one. |
| **MINPS/MAXPS/MINPD/MAXPD** | The definition is `SRC1 < SRC2 ? SRC1 : SRC2`, and the `else` carries both the NaN case and the tie — which is how MINPD tells `+0.0` from `-0.0`. Testing `b < a` gets both wrong. |
| **SHLD/SHRD with a count of zero** | Nothing shifts and no flag changes, but the destination register is still written, and a 32-bit write zeroes the upper half. Returning early left the old upper half in place. |
| **SHLD/SHRD's OF** | Defined for a count of one, and unimplemented. |

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

## Speed, honestly

The emulator is an interpreter, and this is what that costs:

| | native | `qemu-x86_64` | `x86emu` | `x86emu` in wasm |
| --- | --- | --- | --- | --- |
| runtime comes up (`probe`) | — | — | 4 s | 13 s |
| dictionary load + text analysis | 0.0 s | — | 1.0 s | 1.0 s |
| model decrypt + session init | 0.6 s | 6.4 s | 8-9 min | ~25 min |
| `tts`, 0.52 s of audio | 0.5 s | — | 42 min | — |
| `tts`, 1.45 s of audio | 1.5 s | 517 s | 118 min | — |

And the audio that comes out is the real thing. Against a native run of the same
text, `tools/wavcmp.mjs` puts the largest sample difference at 3 against a peak
of 3912 for one, and 7 against 12988 for the other - under a tenth of a percent,
which is `RSQRTPS` and `RCPPS` being *approximate* instructions that hardware
answers to twelve bits and this emulator computes exactly.

Two things make the gap that large. The interpreter retires tens of millions of
instructions a second where the machine does billions; and CPUID here advertises
SSE2 only, so ORT's kernels take their slowest path — which is also true of qemu,
and is why qemu's own numbers are nothing like native.

A one-entry-per-slot page cache in `Memory::host_ptr` (this repo's copy of the
emulator has it) took the worst of the hash-table lookups out. Seconds rather
than minutes needs a JIT, which is a different project.

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

- a C++17 compiler for the emulator (MSVC 2022 and gcc both build it)
- to build the guests: WSL's gcc, or clang with `ld.lld` (set `CLANG=`)
- to build the browser demo: emscripten (set `EMCC=`)

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
