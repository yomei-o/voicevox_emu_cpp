# voicevox_emu_cpp

The official VOICEVOX CORE, running inside an emulator.

VOICEVOX's released voice models are encrypted. They are `vv_bin` payloads that
only [`voicevox_onnxruntime`](https://github.com/VOICEVOX/onnxruntime-builder) —
a patched ONNX Runtime whose source repository is private — can open. That is
why [voicevox_core_cpp](../voicevox_core_cpp), a from-scratch reimplementation
with its own ONNX engine, can run the whole pipeline and still not say a word:
there is no plaintext model for it to run.

This project takes the other road. Rather than reimplement the runtime, it
**runs the official binaries as they are**, one x86-64 instruction at a time,
under [x86_emu_cpp](../x86_emu_cpp). Nothing is decompiled and nothing is
extracted: the DLL does its own work, in its own address space, exactly as its
authors intended — only the CPU underneath it is software.

Which makes the interesting property portability rather than speed. The guest
is a Linux x86-64 program; the host is whatever the emulator builds for, which
today includes Windows on ARM and a browser tab.

## State

**`probe` passes with an unmodified emulator.** The real
`ld-linux-x86-64.so.2` maps the 18 MB runtime and its libstdc++, IFUNC
dispatch settles on the SSE2 string routines, the C++ static initialisers run,
and ORT comes up:

```console
$ sh run_probe.sh
ok    dlopen
ok    dlsym OrtGetApiBase
ok    OrtGetApiBase -> version 1.17.3
ok    GetApi(17)
ok    CreateEnv
...
ok    session.use_vv_bin=1
      provider[0] = CPUExecutionProvider
      tensor[0..5] = 1 2 3 4 5 6
PROBE OK
```

**A plain `.onnx` loads and builds a session** through the emulated runtime —
`predict_duration.onnx` comes up with its real input and output names. ORT's
protobuf parser, graph builder and session init are all sound in here.

**`tts` reaches `voicevox_synthesizer_load_voice_model` and stops there.** The
encrypted model does not decrypt. Two emulator gaps were found and fixed on the
way (`statx`, and AES-NI with its FIPS-197 test), neither of which was the
cause. `resume.md` has the four controls that corner it and where to look next.

## Getting there

    sh setup.sh          # unpack, build the emulator, build the guests
    sh run_probe.sh
    sh run_tts.sh

**A clone is self-contained.** The runtime, the core, ずんだもん's voice model,
the Open JTalk dictionary and a Debian sysroot are all here; `setup.sh`
downloads nothing. See `licenses/README.md` for what is redistributed under
what terms — all of it permits redistribution, and the voice models require a
credit (`VOICEVOX:ずんだもん`).

No Linux host and no WSL is involved at any point: the sysroot is unpacked
Debian amd64 packages, so an **ARM64 Windows** machine builds and runs an
x86-64 Linux guest. Which is what this was developed on.

## Layout

| | |
| --- | --- |
| `src/probe.c` | does the runtime load and initialise at all? Optionally loads a plain `.onnx`. |
| `src/tts.c` | the whole thing: text in, `out.wav` out, timed per step. Also builds for Windows, as the control. |
| `setup.sh` | unpack, build the emulator, build the guests |
| `unpack.sh` | expands `guest/open_jtalk_dic_utf_8-1.11.tar.gz` (23 MB committed, 107 MB on disk) |
| `sysroot/`, `guest/` | the committed payload — Debian glibc/libstdc++, and the VOICEVOX binaries and model |
| `make_sysroot.sh` | rebuilds `sysroot/` from Debian bookworm `.deb`s. Not needed for a clone. |
| `fetch_models.sh` | re-fetches the runtime, core, a different voice model, the dictionary. Not needed for a clone. |
| `build.sh` | cross-compiles both guests with clang + lld |
| `run_probe.sh`, `run_tts.sh` | run them under the emulator |
| `x86_emu_cpp/` | a copy of the emulator, carrying the `statx` and AES-NI changes |

## Requirements

- clang and `ld.lld` (LLVM 19 was used here) to cross-compile the guests;
  set `CLANG=` if it is not at `/c/Program Files/LLVM/bin/clang.exe`
- `curl`, `tar`, `ar`, `xz` — a git-bash or MSYS shell has all four
- a C++17 compiler, to build the vendored emulator

## Terms

`licenses/README.md` has the full table. The short version: everything
committed here permits redistribution, and audio produced with a VOICEVOX
voice library must be credited — `VOICEVOX:ずんだもん` and the like. See
<https://zunko.jp/con_ongen_kiyaku.html>.

What the voice-model terms *do* forbid is 「逆コンパイル・リバースエンジニア
リング及びこれらの方法の公開すること」. Running the runtime is its intended use
and is not that. Reading a decrypted model back out of guest memory would be,
and this project does not do it — the emulator's `--dump` is pointed at
nothing here for a reason.
