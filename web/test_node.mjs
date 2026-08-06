// Runs the WebAssembly build outside a browser, against the real payload.
//
// The demo page and this test go through the same x86emu.js, the same C entry
// points and the same output callbacks, so everything but the DOM wiring is
// covered here - and a failure prints a stack trace instead of vanishing into
// a worker.
//
//   node web/test_node.mjs isatest          # is the wasm build's ISA right
//   node web/test_node.mjs probe            # does the runtime come up
//   node web/test_node.mjs analyze こんにちは  # text analysis, seconds
//   node web/test_node.mjs tts あ            # the whole thing, and slow
//
// It reads the payload straight out of the working tree, so run setup.sh (or at
// least unpack.sh) first.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');

const what = process.argv[2] || 'probe';
const text = process.argv[3] || 'あ';
const style = process.argv[4] || '3';

const createX86Emu = require(path.join(here, 'x86emu.js'));

const decoder = new TextDecoder('utf-8', { fatal: false });
globalThis.x86emuOutput = (fd, bytes) => process.stdout.write(decoder.decode(bytes, { stream: true }));
globalThis.x86emuLog = (line) => console.error('[emu]', line);

const mod = await createX86Emu();
const SYSROOT = '/sysroot';

function put(guestPath, hostPath) {
    const full = SYSROOT + guestPath;
    mod.FS.mkdirTree(path.posix.dirname(full));
    mod.FS.writeFile(full, new Uint8Array(fs.readFileSync(hostPath)));
}

function putTree(guestDir, hostDir) {
    for (const name of fs.readdirSync(hostDir)) {
        const h = path.join(hostDir, name);
        if (fs.statSync(h).isDirectory()) putTree(guestDir + '/' + name, h);
        else put(guestDir + '/' + name, h);
    }
}

console.error('== filling the guest filesystem');
const libs = [
    ['/lib64/ld-linux-x86-64.so.2', 'sysroot/lib64/ld-linux-x86-64.so.2'],
    ['/lib/x86_64-linux-gnu/libc.so.6', 'sysroot/lib/x86_64-linux-gnu/libc.so.6'],
    ['/lib/x86_64-linux-gnu/libm.so.6', 'sysroot/lib/x86_64-linux-gnu/libm.so.6'],
    ['/lib/x86_64-linux-gnu/libdl.so.2', 'sysroot/lib/x86_64-linux-gnu/libdl.so.2'],
    ['/lib/x86_64-linux-gnu/libpthread.so.0', 'sysroot/lib/x86_64-linux-gnu/libpthread.so.0'],
    ['/lib/x86_64-linux-gnu/libgcc_s.so.1', 'sysroot/lib/x86_64-linux-gnu/libgcc_s.so.1'],
    ['/usr/lib/x86_64-linux-gnu/libstdc++.so.6', 'sysroot/usr/lib/x86_64-linux-gnu/libstdc++.so.6'],
    ['/proc/cpuinfo', 'sysroot/proc/cpuinfo'],
    ['/opt/vv/libvoicevox_core.so', 'guest/libvoicevox_core.so'],
    ['/opt/vv/libvoicevox_onnxruntime.so.1.17.3', 'guest/libvoicevox_onnxruntime.so.1.17.3'],
];
for (const [g, h] of libs) put(g, path.join(root, h));
put('/opt/vv/' + what, path.join(root, 'guest', what));
if (what === 'analyze') {
    putTree('/opt/vv/open_jtalk_dic_utf_8-1.11',
            path.join(root, 'guest/open_jtalk_dic_utf_8-1.11'));
}
if (what === 'tts') {
    put('/opt/vv/0.vvm', path.join(root, 'guest/0.vvm'));
    putTree('/opt/vv/open_jtalk_dic_utf_8-1.11',
            path.join(root, 'guest/open_jtalk_dic_utf_8-1.11'));
}

mod.ccall('emu_set_sysroot', null, ['string'], [SYSROOT]);

const args =
    what === 'tts'
        ? ['/opt/vv/tts', '/opt/vv/libvoicevox_onnxruntime.so.1.17.3',
           '/opt/vv/open_jtalk_dic_utf_8-1.11', '/opt/vv/0.vvm', text, style, '/opt/vv/out.wav']
        : what === 'analyze'
            ? ['/opt/vv/analyze', '/opt/vv/open_jtalk_dic_utf_8-1.11', text]
            : what === 'isatest'
                ? ['/opt/vv/isatest']
                : ['/opt/vv/probe', '/opt/vv/libvoicevox_onnxruntime.so.1.17.3'];

const enc = new TextEncoder();
const parts = args.map((a) => enc.encode(a));
const argv = new Uint8Array(parts.reduce((n, p) => n + p.length + 1, 0));
let at = 0;
for (const p of parts) { argv.set(p, at); at += p.length + 1; }

console.error(`== running ${args[0]}`);
const t0 = Date.now();
const rc = mod.ccall('emu_run_path', 'number',
                     ['string', 'array', 'number', 'number', 'number'],
                     [SYSROOT + '/opt/vv/' + what, argv, argv.length, 0, 0]);
const secs = (Date.now() - t0) / 1000;
const insns = mod.ccall('emu_instructions', 'number', [], []);
console.error(`\n== exit ${rc} in ${secs.toFixed(1)} s (${(insns / 1e9).toFixed(2)} G instructions)`);
if (rc !== 0) {
    console.error('error:', mod.ccall('emu_error', 'string', [], []));
    process.exit(1);
}
if (what === 'tts') {
    const wav = mod.FS.readFile(SYSROOT + '/opt/vv/out.wav');
    const out = path.join(root, 'web_out.wav');
    fs.writeFileSync(out, Buffer.from(wav));
    console.error(`wrote ${out}, ${wav.length} bytes`);
}
