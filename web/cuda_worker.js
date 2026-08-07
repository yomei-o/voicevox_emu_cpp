// The worker behind web/cuda.html.
//
// Same shape as worker.js and a different module: x86emu_cuda.js carries the
// CUDA shim, so the guest's kernel launches are answered as compiled
// WebAssembly instead of being interpreted.  Synthesis goes from hours to
// about twenty seconds; building the sessions still takes the better part of
// fifteen minutes, and that is now the whole cost.
//
// The two CUDA libraries are not in this repository, so the page asks for them
// rather than fetching them.  Everything else comes from the same place the
// ordinary demo gets it.
importScripts('untar.js');
importScripts('x86emu_cuda.js');

// The guest paths are the ones tools/wslrun_cuda.sh uses, deliberately.  A
// saved session records the path it knew each file by, so a state built here
// resumes there and one built there resumes here - which is the difference
// between "the browser can save a session" and "the browser can save a session
// only it can read".
const SYSROOT = '/sysroot';
const STATE = '/state/session';
let Module = null;
let runPath = null;
let resumePath = null;
let setSysroot = null;
let setEnv = null;
let guestEnv = null;
let ready = false;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const status = (text) => post({ type: 'status', text });
const out = (text) => post({ type: 'out', text });

async function fetchBytes(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(url + ': ' + res.status);
    if (!res.body) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        if (onProgress) onProgress(got);
    }
    const all = new Uint8Array(got);
    let at = 0;
    for (const c of chunks) { all.set(c, at); at += c.length; }
    return all;
}

async function gunzip(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

function writeInto(guestPath, data) {
    const full = SYSROOT + guestPath;
    const dir = full.slice(0, full.lastIndexOf('/'));
    if (dir) Module.FS.mkdirTree(dir);
    Module.FS.writeFile(full, data);
}

async function startModule() {
    if (Module) return;
    status('エミュレータを起動しています');
    Module = await createX86EmuCuda();
    runPath = Module.cwrap('emu_run_path', 'number',
                           ['string', 'array', 'number', 'number', 'number']);
    resumePath = Module.cwrap('emu_resume_path', 'number',
                              ['string', 'array', 'number', 'number', 'number', 'string']);
    setSysroot = Module.cwrap('emu_set_sysroot', null, ['string']);
    // Through the C library, because emscripten builds `environ` as the runtime
    // starts and never reads a later write to Module.ENV.  Without this the
    // shim's diagnostics are silently off, which looks like having nothing to
    // report.
    setEnv = Module.cwrap('emu_setenv', null, ['string', 'string']);
    // And the guest's own, which is a different environment entirely.
    guestEnv = Module.cwrap('emu_guest_setenv', null, ['string', 'string']);

    const decoder = new TextDecoder('utf-8', { fatal: false });
    globalThis.x86emuOutput = (fd, bytes) => out(decoder.decode(bytes, { stream: true }));
    globalThis.x86emuLog = (line) => out('[emu] ' + line + '\n');

    try { Module.FS.mkdirTree(SYSROOT + '/tmp'); } catch (e) { /* already there */ }
    setSysroot(SYSROOT);
}

// What the page fetches for itself.  The CUDA build is not among them.
const FILES = [
    ['/lib64/ld-linux-x86-64.so.2', '../sysroot/lib64/ld-linux-x86-64.so.2'],
    ['/lib/x86_64-linux-gnu/libc.so.6', '../sysroot/lib/x86_64-linux-gnu/libc.so.6'],
    ['/lib/x86_64-linux-gnu/libm.so.6', '../sysroot/lib/x86_64-linux-gnu/libm.so.6'],
    ['/lib/x86_64-linux-gnu/libdl.so.2', '../sysroot/lib/x86_64-linux-gnu/libdl.so.2'],
    ['/lib/x86_64-linux-gnu/libpthread.so.0', '../sysroot/lib/x86_64-linux-gnu/libpthread.so.0'],
    ['/lib/x86_64-linux-gnu/libgcc_s.so.1', '../sysroot/lib/x86_64-linux-gnu/libgcc_s.so.1'],
    ['/usr/lib/x86_64-linux-gnu/libstdc++.so.6', '../sysroot/usr/lib/x86_64-linux-gnu/libstdc++.so.6'],
    ['/proc/cpuinfo', '../sysroot/proc/cpuinfo'],
    // cudavvm names libvoicevox_core.so with no RUNPATH, so it has to be where
    // a loader looks - beside the program is a shell's idea, not a loader's.
    ['/lib/x86_64-linux-gnu/libvoicevox_core.so', '../guest/libvoicevox_core.so'],
    ['/opt/vvcuda/cudavvm', 'guest/cudavvm', 493],
    ['/opt/vvcuda/0.vvm', '../guest/0.vvm'],
    // The forwarding stand-ins: SSE2, and they hand every kernel to the host.
    ['/lib/x86_64-linux-gnu/libcudart.so.12', 'guest/cudaguest/libcudart.so.12'],
    ['/lib/x86_64-linux-gnu/libcublas.so.12', 'guest/cudaguest/libcublas.so.12'],
    ['/lib/x86_64-linux-gnu/libcublasLt.so.12', 'guest/cudaguest/libcublasLt.so.12'],
    ['/lib/x86_64-linux-gnu/libcudnn.so.8', 'guest/cudaguest/libcudnn.so.8'],
    ['/lib/x86_64-linux-gnu/libcufft.so.11', 'guest/cudaguest/libcufft.so.11'],
];

// The CUDA runtime, gzipped, because the slimmed provider is still 439 MB on
// disk and 8.8 MB compressed.  Fetched like everything else; the file picker is
// now an override rather than a requirement.
const CUDA_FILES = [
    ['/opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3',
     '../guest/cuda/libvoicevox_onnxruntime.so.1.17.3.gz'],
    ['/opt/vvcuda/libvoicevox_onnxruntime_providers_shared.so',
     '../guest/cuda/libvoicevox_onnxruntime_providers_shared.so.gz'],
    ['/opt/vvcuda/libvoicevox_onnxruntime_providers_cuda.so',
     '../guest/cuda/libvoicevox_onnxruntime_providers_cuda.so.gz'],
];

// A file the page was handed instead.  The names are matched loosely: a file
// picker gives back whatever was selected, and the three are told apart by what
// is in them.
function classify(name) {
    if (/providers_cuda/.test(name)) return '/opt/vvcuda/libvoicevox_onnxruntime_providers_cuda.so';
    if (/providers_shared/.test(name)) return '/opt/vvcuda/libvoicevox_onnxruntime_providers_shared.so';
    if (/libvoicevox_onnxruntime\.so/.test(name)) return '/opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3';
    return null;
}

async function prepare(supplied) {
    await startModule();
    if (ready) return;

    // Anything the page was handed wins over what the repository carries: a
    // newer runtime, or the real provider instead of the slimmed one.
    const given = new Set();
    for (const file of supplied) {
        const where = classify(file.name);
        if (!where) {
            out('無視: ' + file.name + ' (名前から用途が分かりません)\n');
            continue;
        }
        status('読み込み: ' + file.name);
        writeInto(where, new Uint8Array(await file.arrayBuffer()));
        given.add(where);
    }

    let done = 0;
    const total = FILES.length + CUDA_FILES.length + 1;
    for (const [guest, url] of CUDA_FILES) {
        if (given.has(guest)) { done++; continue; }
        status(`ダウンロード (${++done}/${total}): ${guest.split('/').pop()}`);
        post({ type: 'progress', done, total });
        writeInto(guest, await gunzip(await fetchBytes(url)));
    }
    for (const [guest, url, mode] of FILES) {
        status(`ダウンロード (${++done}/${total}): ${guest}`);
        post({ type: 'progress', done, total });
        writeInto(guest, await fetchBytes(url));
        if (mode) { try { Module.FS.chmod(SYSROOT + guest, mode); } catch (e) { /* MEMFS */ } }
    }
    status(`ダウンロード (${total}/${total}): Open JTalk 辞書`);
    const gz = await fetchBytes('../guest/open_jtalk_dic_utf_8-1.11.tar.gz');
    status('辞書を展開しています (103 MB)');
    // Into the *parent*: every entry in the archive is already prefixed with
    // `open_jtalk_dic_utf_8-1.11/`.  Naming the directory here as well put
    // sys.dic one level below where the guest was told to look, and Mecab_load
    // reported a directory it could not open - which is true, and says nothing
    // about why.
    untarBytes(await gunzip(gz), (name, data) => writeInto('/opt/vvcuda/' + name, data));
    post({ type: 'progress', done: total, total });
    ready = true;
}

// The guest's argument list.  It is the same every time, including where the
// text comes from: `@path` means the guest reads the file, so what is said is
// decided after the arguments were fixed - which is what lets a resumed run say
// something the saved one never heard.  argv itself was written into the
// guest's stack when it was loaded and comes back with the rest of its memory,
// so it could not vary anyway.
const ARGS = [
    '/opt/vvcuda/cudavvm',
    '/opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3',
    '/opt/vvcuda/open_jtalk_dic_utf_8-1.11',
    '/opt/vvcuda/0.vvm',
    '3',
    '@/opt/vvcuda/text.txt',
    '/opt/vvcuda/out.wav',
];

function argv() {
    const blob = new TextEncoder().encode(ARGS.join('\0') + '\0');
    return [blob, blob.length - 1];
}

// One run of the guest, from the beginning or from a saved session.
function runGuest(state) {
    const [blob, len] = argv();
    const started = Date.now();
    const program = SYSROOT + '/opt/vvcuda/cudavvm';
    const code = state ? resumePath(program, blob, len, 0, 0, state)
                       : runPath(program, blob, len, 0, 0);
    if (code !== 0) {
        throw new Error('exit ' + code + ': ' +
                        Module.ccall('emu_error', 'string', [], []));
    }
    return {
        seconds: (Date.now() - started) / 1000,
        instructions: Module.ccall('emu_instructions', 'number', [], []),
        heap: Module.HEAPU8.length,
    };
}

function speak(text, state) {
    writeInto('/opt/vvcuda/text.txt', new TextEncoder().encode(text));
    const stats = runGuest(state);
    const wav = Module.FS.readFile(SYSROOT + '/opt/vvcuda/out.wav');
    post({ type: 'wav', wav: wav.buffer, ...stats }, [wav.buffer]);
}

// Build the sessions and stop, so the twenty minutes can be paid once.
//
// The guest asks for the state itself, through VVSNAPSHOT - the same path the
// native build takes.  It halts where it asks, with the sessions built and no
// kernel launched, which is why what a resumed run says is still open.
function build() {
    try { Module.FS.mkdirTree('/state'); } catch (e) { /* already there */ }
    // The *guest's* environment, not this process's.  emu_setenv is the other
    // one, which the emulator and the shim read for their own diagnostics and
    // the guest cannot see - setting VVSNAPSHOT there ran the whole thing
    // through to synthesis without ever taking a snapshot.
    guestEnv('VVSNAPSHOT', STATE);
    writeInto('/opt/vvcuda/text.txt', new TextEncoder().encode('あ'));
    const stats = runGuest(null);
    guestEnv('VVSNAPSHOT', '');
    // Handed over as Blobs rather than as bytes.  A Blob crosses by reference,
    // so the page can turn it into a download without a second 166 MB of it
    // existing: MEMFS already holds one, and the WebAssembly heap is near a
    // gigabyte by this point.
    const session = new Blob([Module.FS.readFile(STATE)],
                             { type: 'application/octet-stream' });
    const shim = new Blob([Module.FS.readFile(STATE + '.shim')],
                          { type: 'application/octet-stream' });
    post({ type: 'state', session, shim, ...stats });
}

// A session from a file the page was given.  Written where the emulator opens
// it as a host path, outside the sysroot the guest can see.
function loadState(session, shim) {
    try { Module.FS.mkdirTree('/state'); } catch (e) { /* already there */ }
    Module.FS.writeFile(STATE, new Uint8Array(session));
    Module.FS.writeFile(STATE + '.shim', new Uint8Array(shim));
}

self.onmessage = async (e) => {
    try {
        if (e.data.type === 'speak') {
            await prepare(e.data.files || []);
            status('合成しています - セッション構築に十数分、そのあと合成は数十秒');
            speak(e.data.text || 'ずんだもんなのだ', null);
        } else if (e.data.type === 'build') {
            await prepare(e.data.files || []);
            status('セッションを構築しています - 十数分かかります');
            build();
        } else if (e.data.type === 'resume') {
            await prepare(e.data.files || []);
            status('セッションを読み込んでいます');
            loadState(e.data.session, e.data.shim);
            status('話しています');
            speak(e.data.text || 'ずんだもんなのだ', STATE);
        } else if (e.data.type === 'again') {
            status('話しています');
            speak(e.data.text || 'ずんだもんなのだ', STATE);
        }
    } catch (err) {
        post({ type: 'error', text: String(err && err.message ? err.message : err) });
    }
};
