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

const SYSROOT = '/sysroot';
let Module = null;
let runPath = null;
let setSysroot = null;
let setEnv = null;
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
    setSysroot = Module.cwrap('emu_set_sysroot', null, ['string']);
    // Through the C library, because emscripten builds `environ` as the runtime
    // starts and never reads a later write to Module.ENV.  Without this the
    // shim's diagnostics are silently off, which looks like having nothing to
    // report.
    setEnv = Module.cwrap('emu_setenv', null, ['string', 'string']);

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
    ['/opt/vv/cudavvm', 'guest/cudavvm', 493],
    ['/opt/vv/0.vvm', '../guest/0.vvm'],
    // The forwarding stand-ins: SSE2, and they hand every kernel to the host.
    ['/lib/x86_64-linux-gnu/libcudart.so.12', 'guest/cudaguest/libcudart.so.12'],
    ['/lib/x86_64-linux-gnu/libcublas.so.12', 'guest/cudaguest/libcublas.so.12'],
    ['/lib/x86_64-linux-gnu/libcublasLt.so.12', 'guest/cudaguest/libcublasLt.so.12'],
    ['/lib/x86_64-linux-gnu/libcudnn.so.8', 'guest/cudaguest/libcudnn.so.8'],
    ['/lib/x86_64-linux-gnu/libcufft.so.11', 'guest/cudaguest/libcufft.so.11'],
];

// What the page has to be given, because this repository does not carry it.
// The names are matched loosely: a file picker hands back whatever the user
// selected, and the three are told apart by what is in them.
function classify(name) {
    if (/providers_cuda/.test(name)) return '/opt/vv/libvoicevox_onnxruntime_providers_cuda.so';
    if (/providers_shared/.test(name)) return '/opt/vv/libvoicevox_onnxruntime_providers_shared.so';
    if (/libvoicevox_onnxruntime\.so/.test(name)) return '/opt/vv/libvoicevox_onnxruntime.so.1.17.3';
    return null;
}

async function prepare(supplied) {
    await startModule();
    if (ready) return;

    const need = new Set(['/opt/vv/libvoicevox_onnxruntime.so.1.17.3',
                          '/opt/vv/libvoicevox_onnxruntime_providers_shared.so',
                          '/opt/vv/libvoicevox_onnxruntime_providers_cuda.so']);
    for (const file of supplied) {
        const where = classify(file.name);
        if (!where) {
            out('無視: ' + file.name + ' (名前から用途が分かりません)\n');
            continue;
        }
        status('読み込み: ' + file.name);
        writeInto(where, new Uint8Array(await file.arrayBuffer()));
        need.delete(where);
    }
    if (need.size) {
        throw new Error('足りません: ' + [...need].map((p) => p.split('/').pop()).join(', '));
    }

    let done = 0;
    const total = FILES.length + 1;
    for (const [guest, url, mode] of FILES) {
        status(`ダウンロード (${++done}/${total}): ${guest}`);
        post({ type: 'progress', done, total });
        writeInto(guest, await fetchBytes(url));
        if (mode) { try { Module.FS.chmod(SYSROOT + guest, mode); } catch (e) { /* MEMFS */ } }
    }
    status(`ダウンロード (${total}/${total}): Open JTalk 辞書`);
    const gz = await fetchBytes('../guest/open_jtalk_dic_utf_8-1.11.tar.gz');
    status('辞書を展開しています (103 MB)');
    untarBytes(await gunzip(gz),
               (name, data) => writeInto('/opt/vv/open_jtalk_dic_utf_8-1.11/' + name, data));
    post({ type: 'progress', done: total, total });
    ready = true;
}

function speak(text) {
    writeInto('/opt/vv/text.txt', new TextEncoder().encode(text));
    const args = [
        '/opt/vv/cudavvm',
        '/opt/vv/libvoicevox_onnxruntime.so.1.17.3',
        '/opt/vv/open_jtalk_dic_utf_8-1.11',
        '/opt/vv/0.vvm',
        '3',
        '@/opt/vv/text.txt',
        '/opt/vv/out.wav',
    ];
    const blob = new TextEncoder().encode(args.join('\0') + '\0');
    const started = Date.now();
    const code = runPath(SYSROOT + '/opt/vv/cudavvm', blob, blob.length - 1, 0, 0);
    const seconds = (Date.now() - started) / 1000;
    if (code !== 0) {
        throw new Error('exit ' + code + ': ' +
                        Module.ccall('emu_error', 'string', [], []));
    }
    const wav = Module.FS.readFile(SYSROOT + '/opt/vv/out.wav');
    post({
        type: 'wav',
        wav: wav.buffer,
        seconds,
        instructions: Module.ccall('emu_instructions', 'number', [], []),
        heap: Module.HEAPU8.length,
    }, [wav.buffer]);
}

self.onmessage = async (e) => {
    try {
        if (e.data.type === 'speak') {
            await prepare(e.data.files || []);
            status('合成しています - セッション構築に十数分、そのあと合成は数十秒');
            speak(e.data.text || 'ずんだもんなのだ');
        }
    } catch (err) {
        post({ type: 'error', text: String(err && err.message ? err.message : err) });
    }
};
