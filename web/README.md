# web/ — the emulator as WebAssembly, and the demo page

The same x86-64 emulator this project uses on the desktop, compiled to
WebAssembly, running the same official VOICEVOX guest in a browser tab.

    sh web/build.sh                     # emcc -> web/x86emu.js (wasm embedded)
    node web/test_node.mjs isatest      # is the wasm build's arithmetic right?
    node web/test_node.mjs analyze こんにちは
    node web/test_page.mjs              # the page's own path, without a browser
    node ../tools/browser_test.mjs      # the page, in a headless browser

Serve the repository root over HTTP and open `web/`. It cannot be opened as a
`file://` URL: the page fetches its payload, and a worker needs an origin.

## What is here

| | |
| --- | --- |
| `x86emu.js` | the emulator, wasm embedded (`SINGLE_FILE`), so `web/` is plain static files — no MIME setup, and GitHub Pages serves it as is |
| `wasm_api.cpp` | the entry points: `emu_run_path`, `emu_set_sysroot`, and the output callbacks |
| `build.sh` | builds the above with emscripten |
| `index.html` | the demo page |
| `worker.js` | fetching, unpacking and running, off the main thread |
| `untar.js` | a tar reader, shared with the node rehearsal |
| `payload.json` | what goes into the guest's filesystem, and where it comes from |
| `guest/` | the two guest programs the page runs, prebuilt |
| `sample/` | WAVs this project generated, so the page has something to play |
| `test_node.mjs`, `test_page.mjs` | the same code paths, without a browser |

## The payload

`payload.json` points at files **this repository already carries**, so the demo
adds no copies: the seven shared libraries the dynamic loader actually opens
(traced with `X86EMU_TRACE_OPEN=1`, not the whole 43 MB Debian sysroot), CORE,
the ONNX Runtime, the voice model and the Open JTalk dictionary as its original
`.tar.gz`, which the page unpacks with the browser's own `DecompressionStream`.

It is in two groups because the two halves of VOICEVOX cost very different
things. **Text analysis** — 33 MB, answers in about a second, and is the half a
browser can do interactively. **Synthesis** — another 76 MB, and then 85 minutes
or more, because a neural vocoder on an interpreted CPU is what it is. The page
says so, and ships the WAVs rather than pretending otherwise.

Everything the page fetches goes into the Cache API, so a second visit is free.

## Two things that will bite

- **Run `node web/test_node.mjs isatest` after touching the emulator.** A
  WebAssembly build is a different compiler on a different target, and undefined
  behaviour that happens to be right on x86 is not right there. That is exactly
  how the `CVTPS2DQ` bug was found: the conversion was a plain
  `static_cast<int32_t>(double)`, which on x86 compiles to the very instruction
  being emulated and in wasm to a trapping truncate.
- **A synthesis run needs about 700 MB in the tab** — the payload in MEMFS plus
  the guest's own memory. Desktop only.
