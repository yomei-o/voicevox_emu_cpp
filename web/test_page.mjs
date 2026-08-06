// A rehearsal of what the page does, without a browser.
//
// test_node.mjs reads the payload off the disk; this one serves the repository
// over HTTP and goes through payload.json, fetch, gzip and the same tar reader
// the worker uses - so a broken URL, a mis-declared path or a tar member the
// reader skips shows up here rather than in a tab.
//
//   node web/test_page.mjs            # text analysis: about a minute
//   node web/test_page.mjs speak あ    # the whole thing: hours
//
// Everything but the DOM wiring and the Cache API is covered.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { createServer } from 'node:http';
import { gunzipSync } from 'node:zlib';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');

const untar = require(path.join(here, 'untar.js'));
const createX86Emu = require(path.join(here, 'x86emu.js'));

const mode = process.argv[2] === 'speak' ? 'speak' : 'analyze';
const text = process.argv[3] || (mode === 'speak' ? 'あ' : 'ずんだもんなのだ');
const style = process.argv[4] || '3';

// ---- serve the repository, the way GitHub Pages will --------------------

const types = {'.json': 'application/json', '.html': 'text/html', '.js': 'text/javascript'};
const server = createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
    const file = path.join(root, rel);
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        res.writeHead(404).end('not found');
        return;
    }
    res.writeHead(200, {
        'content-type': types[path.extname(file)] || 'application/octet-stream',
        'content-length': fs.statSync(file).size,
    });
    fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/web/`;
console.error(`serving ${root} at ${base}`);

// ---- the same sequence the worker runs ----------------------------------

const SYSROOT = '/sysroot';
const decoder = new TextDecoder('utf-8', { fatal: false });
let guestOut = '';
globalThis.x86emuOutput = (fd, bytes) => {
    const s = decoder.decode(bytes, { stream: true });
    if (fd === 1) guestOut += s;
    process.stdout.write(s);
};
globalThis.x86emuLog = (line) => console.error('[emu]', line);

const mod = await createX86Emu();
mod.ccall('emu_set_sysroot', null, ['string'], [SYSROOT]);

function writeInto(p, data) {
    const full = SYSROOT + (p.startsWith('/') ? p : '/' + p);
    mod.FS.mkdirTree(full.slice(0, full.lastIndexOf('/')));
    mod.FS.writeFile(full, data);
}

const payload = await (await fetch(base + 'payload.json')).json();

async function prepare(name) {
    const group = payload.groups[name];
    for (const f of group.files) {
        const res = await fetch(new URL(f.url, base));
        if (!res.ok) throw new Error(`${f.url}: HTTP ${res.status}`);
        const data = new Uint8Array(await res.arrayBuffer());
        console.error(`  ${f.path}  ${(data.length / 1e6).toFixed(2)} MB`);
        writeInto(f.path, data);
    }
    for (const a of group.archives) {
        const res = await fetch(new URL(a.url, base));
        if (!res.ok) throw new Error(`${a.url}: HTTP ${res.status}`);
        const gz = Buffer.from(await res.arrayBuffer());
        // The browser uses DecompressionStream; node's zlib is the same gzip.
        const tar = new Uint8Array(gunzipSync(gz));
        let n = 0, bytes = 0;
        untar(tar, (name2, data) => {
            writeInto(a.into + '/' + name2, data);
            n++;
            bytes += data.length;
        });
        console.error(`  ${a.into}  ${n} files, ${(bytes / 1e6).toFixed(1)} MB unpacked`);
    }
}

console.error(`== preparing "analyze"`);
await prepare('analyze');
if (mode === 'speak') {
    console.error(`== preparing "speak"`);
    await prepare('speak');
}

const args = mode === 'speak'
    ? [payload.speakProgram, payload.onnxruntime, payload.dictionary, payload.model,
       text, style, '/opt/vv/out.wav']
    : [payload.analyzeProgram, payload.dictionary, text];

const enc = new TextEncoder();
const parts = args.map((a) => enc.encode(a));
const argv = new Uint8Array(parts.reduce((n, p) => n + p.length + 1, 0));
let at = 0;
for (const p of parts) { argv.set(p, at); at += p.length + 1; }

console.error(`== running ${args[0]} "${text}"`);
const t0 = Date.now();
const rc = mod.ccall('emu_run_path', 'number',
                     ['string', 'array', 'number', 'number', 'number'],
                     [SYSROOT + args[0], argv, argv.length, 0, 0]);
const secs = (Date.now() - t0) / 1000;
const insns = mod.ccall('emu_instructions', 'number', [], []);
console.error(`\n== exit ${rc} in ${secs.toFixed(1)} s (${(insns / 1e9).toFixed(3)} G instructions)`);
server.close();

if (rc !== 0) {
    console.error('error:', mod.ccall('emu_error', 'string', [], []));
    process.exit(1);
}
if (mode === 'speak') {
    const wav = mod.FS.readFile(SYSROOT + '/opt/vv/out.wav');
    fs.writeFileSync(path.join(root, 'web_out.wav'), Buffer.from(wav));
    console.error(`wrote web_out.wav, ${wav.length} bytes`);
} else {
    const phrases = JSON.parse(guestOut.trim());
    console.error(`parsed ${phrases.length} accent phrases, ` +
                  `${phrases.reduce((n, p) => n + p.moras.length, 0)} moras`);
}
console.error('PAGE REHEARSAL OK');
