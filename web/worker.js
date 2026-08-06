// The worker that runs VOICEVOX CORE.
//
// Everything expensive happens here so the page stays responsive: fetching the
// payload, filling the emulated filesystem, and then running an x86-64 Linux
// program one instruction at a time for as long as it takes.
//
// Messages in:   {type:'prepare', group}   {type:'run', mode, text, style}
// Messages out:  {type:'status'|'progress'|'out'|'ready'|'done'|'error', ...}

importScripts('untar.js');
importScripts('x86emu.js');

const SYSROOT = '/sysroot';

let Module = null;
let payload = null;
let runPath = null;
let setSysroot = null;
const loaded = new Set();

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const status = (text) => post({type: 'status', text});

// ---------------------------------------------------------------------------
// fetching, with the payload kept in the Cache API so a second visit is free.
// The cache is missing or refused in some contexts (an insecure origin, private
// browsing), so every use of it may fail and fall back to the network.

const CACHE_NAME = 'voicevox-emu-payload-v1';
let cachePromise;

async function openCache() {
    if (cachePromise === undefined)
        cachePromise = typeof caches === 'undefined' ? Promise.resolve(null)
                                                     : caches.open(CACHE_NAME);
    try {
        return await cachePromise;
    } catch (e) {
        return null;
    }
}

async function readStream(res, onProgress) {
    if (!res.body) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
        const {done, value} = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        if (onProgress) onProgress(got);
    }
    const out = new Uint8Array(got);
    let at = 0;
    for (const c of chunks) {
        out.set(c, at);
        at += c.length;
    }
    return out;
}

async function fetchBytes(url, onProgress) {
    const cache = await openCache();
    if (cache) {
        try {
            const hit = await cache.match(url);
            if (hit) {
                const bytes = new Uint8Array(await hit.arrayBuffer());
                if (onProgress) onProgress(bytes.length);
                return bytes;
            }
        } catch (e) { /* fall through to the network */ }
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    if (cache) {
        try {
            await cache.put(url, res.clone());
        } catch (e) { /* over quota: not worth failing the run over */ }
    }
    return readStream(res, onProgress);
}

async function gunzip(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

// ---------------------------------------------------------------------------

function writeInto(path, data) {
    const full = SYSROOT + (path.startsWith('/') ? path : '/' + path);
    const dir = full.slice(0, full.lastIndexOf('/'));
    try {
        Module.FS.mkdirTree(dir);
    } catch (e) { /* already there */ }
    Module.FS.writeFile(full, data);
}

async function startModule() {
    if (Module) return;
    status('エミュレータを起動しています');
    Module = await createX86Emu();
    runPath = Module.cwrap('emu_run_path', 'number',
                           ['string', 'array', 'number', 'number', 'number']);
    setSysroot = Module.cwrap('emu_set_sysroot', null, ['string']);

    // Guest output is bytes; the page decodes it as UTF-8.
    const decoder = new TextDecoder('utf-8', {fatal: false});
    globalThis.x86emuOutput = (fd, bytes) =>
        post({type: 'out', fd, text: decoder.decode(bytes, {stream: true})});
    globalThis.x86emuLog = (line) => post({type: 'out', fd: 2, text: '[emu] ' + line + '\n'});

    // Somewhere for the guest to put a temporary file.  Nothing the demo runs
    // needs one today, but the guest half of the API does - Open JTalk compiles
    // a user dictionary through /tmp - and an absent directory is a confusing
    // way to find that out.
    try {
        Module.FS.mkdirTree(SYSROOT + '/tmp');
    } catch (e) { /* already there */ }

    setSysroot(SYSROOT);
}

async function prepare(name) {
    await startModule();
    if (loaded.has(name)) {
        post({type: 'ready', group: name});
        return;
    }
    const group = payload.groups[name];
    const total = group.files.concat(group.archives).reduce((n, f) => n + (f.bytes || 0), 0);
    let done = 0;

    for (const f of group.files) {
        status(`ダウンロード: ${f.path}`);
        const data = await fetchBytes(f.url, (got) =>
            post({type: 'progress', done: done + got, total}));
        done += data.length;
        writeInto(f.path, data);
        if (f.mode) {
            try {
                Module.FS.chmod(SYSROOT + f.path, f.mode);
            } catch (e) { /* MEMFS does not always care */ }
        }
        post({type: 'progress', done, total});
    }

    for (const a of group.archives) {
        status('ダウンロード: Open JTalk 辞書 (23 MB)');
        const gz = await fetchBytes(a.url, (got) =>
            post({type: 'progress', done: done + got, total}));
        done += gz.length;
        status('辞書を展開しています (103 MB)');
        untarBytes(await gunzip(gz), (name2, data) => writeInto(a.into + '/' + name2, data));
        post({type: 'progress', done, total});
    }

    loaded.add(name);
    status('準備完了');
    post({type: 'ready', group: name});
}

function argvBlob(args) {
    const enc = new TextEncoder();
    const parts = args.map((a) => enc.encode(a));
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length + 1, 0));
    let at = 0;
    for (const p of parts) {
        out.set(p, at);
        at += p.length + 1;  // the gap is already zero
    }
    return out;
}

function execute(program, args) {
    const argv = argvBlob(args);
    const t0 = performance.now();
    const rc = runPath(SYSROOT + program, argv, argv.length, 0, 0);
    const seconds = (performance.now() - t0) / 1000;
    const instructions = Module.ccall('emu_instructions', 'number', [], []);
    if (rc !== 0) {
        const err = Module.ccall('emu_error', 'string', [], []);
        throw new Error(err || `ゲストが ${rc} で終了しました`);
    }
    return {seconds, instructions};
}

function doAnalyze(text) {
    const r = execute(payload.analyzeProgram, [payload.analyzeProgram, payload.dictionary, text]);
    post({type: 'done', mode: 'analyze', ...r});
}

function doSpeak(text, style) {
    const out = '/opt/vv/out.wav';
    const r = execute(payload.speakProgram, [
        payload.speakProgram, payload.onnxruntime, payload.dictionary, payload.model,
        text, String(style), out,
    ]);
    let wav;
    try {
        wav = Module.FS.readFile(SYSROOT + out);
    } catch (e) {
        throw new Error(`${out} が作られませんでした`);
    }
    post({type: 'done', mode: 'speak', wav, ...r}, [wav.buffer]);
}

self.onmessage = async (e) => {
    const msg = e.data;
    try {
        if (msg.type === 'payload') {
            payload = msg.payload;
        } else if (msg.type === 'prepare') {
            await prepare(msg.group);
        } else if (msg.type === 'run') {
            if (!loaded.has(msg.mode === 'speak' ? 'speak' : 'analyze'))
                throw new Error('まだ準備ができていません');
            if (msg.mode === 'speak') doSpeak(msg.text, msg.style);
            else doAnalyze(msg.text);
        }
    } catch (err) {
        post({type: 'error', text: String(err && err.message ? err.message : err)});
    }
};
