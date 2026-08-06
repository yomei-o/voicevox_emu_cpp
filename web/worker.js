// The worker that runs VOICEVOX CORE.
//
// Everything expensive happens here so the page stays responsive: fetching the
// payload, filling the emulated filesystem, and then running an x86-64 Linux
// program one instruction at a time for as long as it takes.
//
// Messages in:   {type:'prepare', payload}   {type:'speak', text, style}
// Messages out:  {type:'status'|'progress'|'out'|'ready'|'done'|'error', ...}

importScripts('x86emu.js');

let Module = null;
let prepared = false;
let run = null;
let setSysroot = null;

const SYSROOT = '/sysroot';

function post(msg) {
    self.postMessage(msg);
}

function status(text) {
    post({type: 'status', text});
}

// ---------------------------------------------------------------------------
// tar, because the Open JTalk dictionary ships as one and unpacking it here
// saves a hundred megabytes of separate requests.

function untar(bytes, write) {
    let off = 0;
    const dec = new TextDecoder();
    const str = (start, len) => {
        const s = bytes.subarray(off + start, off + start + len);
        let end = s.indexOf(0);
        return dec.decode(end < 0 ? s : s.subarray(0, end));
    };
    while (off + 512 <= bytes.length) {
        const name = str(0, 100);
        if (!name) {           // two zero blocks end the archive
            off += 512;
            continue;
        }
        const size = parseInt(str(124, 12).trim() || '0', 8);
        const type = String.fromCharCode(bytes[off + 156]);
        const prefix = str(345, 155);
        const full = prefix ? prefix + '/' + name : name;
        off += 512;
        if (type === '0' || type === '\0') {
            write(full, bytes.subarray(off, off + size));
        }
        off += Math.ceil(size / 512) * 512;
    }
}

async function fetchBytes(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const total = Number(res.headers.get('content-length')) || 0;
    if (!res.body) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
        const {done, value} = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        if (onProgress) onProgress(got, total);
    }
    const out = new Uint8Array(got);
    let at = 0;
    for (const c of chunks) {
        out.set(c, at);
        at += c.length;
    }
    return out;
}

// A gzip stream, decoded by the browser rather than by a library.
async function gunzip(bytes) {
    const ds = new DecompressionStream('gzip');
    const stream = new Blob([bytes]).stream().pipeThrough(ds);
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

function writeInto(path, data) {
    const full = SYSROOT + (path.startsWith('/') ? path : '/' + path);
    const dir = full.slice(0, full.lastIndexOf('/'));
    try {
        Module.FS.mkdirTree(dir);
    } catch (e) { /* already there */ }
    Module.FS.writeFile(full, data);
}

// ---------------------------------------------------------------------------

async function prepare(payload) {
    if (prepared) return;

    status('エミュレータを起動しています');
    Module = await createX86Emu();
    run = Module.cwrap('emu_run_path', 'number',
                       ['string', 'array', 'number', 'number', 'number']);
    setSysroot = Module.cwrap('emu_set_sysroot', null, ['string']);

    // Guest output arrives as raw bytes; the page decodes it as UTF-8.
    const decoder = new TextDecoder('utf-8', {fatal: false});
    globalThis.x86emuOutput = (fd, bytes) => {
        post({type: 'out', fd, text: decoder.decode(bytes, {stream: true})});
    };
    globalThis.x86emuLog = (line) => post({type: 'out', fd: 2, text: '[emu] ' + line + '\n'});

    const items = payload.files.concat(payload.archives);
    const totalBytes = items.reduce((n, f) => n + (f.bytes || 0), 0);
    let doneBytes = 0;

    for (const f of payload.files) {
        status(`ダウンロード: ${f.path}`);
        const data = await fetchBytes(f.url, (got) => {
            post({type: 'progress', done: doneBytes + got, total: totalBytes});
        });
        doneBytes += data.length;
        writeInto(f.path, data);
        if (f.mode) Module.FS.chmod(SYSROOT + f.path, f.mode);
        post({type: 'progress', done: doneBytes, total: totalBytes});
    }

    for (const a of payload.archives) {
        status(`ダウンロード: ${a.into} (辞書)`);
        const gz = await fetchBytes(a.url, (got) => {
            post({type: 'progress', done: doneBytes + got, total: totalBytes});
        });
        doneBytes += gz.length;
        status('辞書を展開しています');
        const tar = await gunzip(gz);
        untar(tar, (name, data) => writeInto(a.into + '/' + name, data));
        post({type: 'progress', done: doneBytes, total: totalBytes});
    }

    setSysroot(SYSROOT);
    prepared = true;
    status('準備完了');
    post({type: 'ready'});
}

function speak(payload, text, style, outPath) {
    // argv is a NUL-separated blob, so the text goes through as UTF-8 with
    // nothing in between to mangle it.
    const args = [
        payload.program,
        '/opt/vv/' + payload.onnxruntime,
        '/opt/vv/' + payload.dictionary,
        '/opt/vv/' + payload.model,
        text,
        String(style),
        outPath,
    ];
    const enc = new TextEncoder();
    const parts = args.map((a) => enc.encode(a));
    const len = parts.reduce((n, p) => n + p.length + 1, 0);
    const argv = new Uint8Array(len);
    let at = 0;
    for (const p of parts) {
        argv.set(p, at);
        at += p.length + 1;  // the gap is already zero
    }

    const t0 = performance.now();
    const rc = run(SYSROOT + payload.program, argv, argv.length, 0, 0);
    const seconds = (performance.now() - t0) / 1000;

    if (rc !== 0) {
        const err = Module.ccall('emu_error', 'string', [], []);
        post({type: 'error', text: err || `ゲストが ${rc} で終了しました`, seconds});
        return;
    }
    let wav = null;
    try {
        wav = Module.FS.readFile(SYSROOT + outPath);
    } catch (e) {
        post({type: 'error', text: `${outPath} が作られませんでした`, seconds});
        return;
    }
    const instructions = Module.ccall('emu_instructions', 'number', [], []);
    post({type: 'done', wav, seconds, instructions}, [wav.buffer]);
}

let payloadCache = null;

self.onmessage = async (e) => {
    const msg = e.data;
    try {
        if (msg.type === 'prepare') {
            payloadCache = msg.payload;
            await prepare(msg.payload);
        } else if (msg.type === 'speak') {
            if (!prepared) throw new Error('まだ準備ができていません');
            speak(payloadCache, msg.text, msg.style, msg.out || '/opt/vv/out.wav');
        }
    } catch (err) {
        post({type: 'error', text: String(err && err.message ? err.message : err)});
    }
};
