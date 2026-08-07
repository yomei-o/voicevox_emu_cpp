// The CUDA path, in WebAssembly, measured.
//
//   node web/test_cuda.mjs [text]
//
// What it is for: the browser number for this path has only ever been a
// calculation - 7.8 G instructions divided by the six million a second the
// ordinary demo retires.  This runs it.
//
// It needs, beside the usual sysroot:
//   the CUDA build of voicevox_onnxruntime and its two provider libraries
//   guest/cudaguest/*.so.*   the SSE2 stand-ins that forward to the host
//   guest/cudavvm            the program
//
// VVCUDA=<dir> says where the CUDA build is; the slimmed provider is preferred
// if one is beside it, because the original is 460 MB of which 419 is device
// code nothing here executes.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');
const text = process.argv[2] || 'あ';
const cuda = process.env.VVCUDA ||
    path.join(process.env.HOME || '', 'vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib');

const createX86EmuCuda = require(path.join(here, 'x86emu_cuda.js'));

const decoder = new TextDecoder('utf-8', { fatal: false });
globalThis.x86emuOutput = (fd, bytes) => process.stdout.write(decoder.decode(bytes, { stream: true }));
globalThis.x86emuLog = (line) => console.error('[emu]', line);

const mod = await createX86EmuCuda();
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
function mb(n) { return (n / 1048576).toFixed(1) + ' MB'; }

console.error('== filling the guest filesystem');
for (const [guest, host] of [
    ['/lib64/ld-linux-x86-64.so.2', 'sysroot/lib64/ld-linux-x86-64.so.2'],
    ['/lib/x86_64-linux-gnu/libc.so.6', 'sysroot/lib/x86_64-linux-gnu/libc.so.6'],
    ['/lib/x86_64-linux-gnu/libm.so.6', 'sysroot/lib/x86_64-linux-gnu/libm.so.6'],
    ['/lib/x86_64-linux-gnu/libdl.so.2', 'sysroot/lib/x86_64-linux-gnu/libdl.so.2'],
    ['/lib/x86_64-linux-gnu/libpthread.so.0', 'sysroot/lib/x86_64-linux-gnu/libpthread.so.0'],
    ['/lib/x86_64-linux-gnu/libgcc_s.so.1', 'sysroot/lib/x86_64-linux-gnu/libgcc_s.so.1'],
    ['/usr/lib/x86_64-linux-gnu/libstdc++.so.6', 'sysroot/usr/lib/x86_64-linux-gnu/libstdc++.so.6'],
    ['/proc/cpuinfo', 'sysroot/proc/cpuinfo'],
]) put(guest, path.join(root, host));

// The stand-ins go where a Debian with CUDA installed would keep them, which is
// where the provider's own search looks - the guest is given only PATH, so
// LD_LIBRARY_PATH is not an option.
putTree('/lib/x86_64-linux-gnu', path.join(root, 'guest/cudaguest'));

put('/opt/vv/cudavvm', path.join(root, 'guest/cudavvm'));
// cudavvm names libvoicevox_core.so with no RUNPATH, so ld.so looks only where
// it looks for any library.  Being beside the program is not one of those
// places - that is a shell's idea, not a loader's.
put('/lib/x86_64-linux-gnu/libvoicevox_core.so', path.join(root, 'guest/libvoicevox_core.so'));
put('/opt/vv/0.vvm', path.join(root, 'guest/0.vvm'));
put('/opt/vv/text.txt', (() => {
    const t = path.join(root, 'build/wasm_text.txt');
    fs.mkdirSync(path.dirname(t), { recursive: true });
    fs.writeFileSync(t, text);
    return t;
})());
putTree('/opt/vv/open_jtalk_dic_utf_8-1.11', path.join(root, 'guest/open_jtalk_dic_utf_8-1.11'));

put('/opt/vv/libvoicevox_onnxruntime.so.1.17.3',
    path.join(cuda, 'libvoicevox_onnxruntime.so.1.17.3'));
put('/opt/vv/libvoicevox_onnxruntime_providers_shared.so',
    path.join(cuda, 'libvoicevox_onnxruntime_providers_shared.so'));

const slim = path.join(cuda, 'libvoicevox_onnxruntime_providers_cuda_slim.so');
const provider = fs.existsSync(slim)
    ? slim : path.join(cuda, 'libvoicevox_onnxruntime_providers_cuda.so');
console.error('   provider ' + mb(fs.statSync(provider).size) +
              (provider === slim ? ' (slimmed)' : ' (as shipped)'));
put('/opt/vv/libvoicevox_onnxruntime_providers_cuda.so', provider);

// Two different memories, and the difference is the whole browser question.
// MEMFS keeps file contents in JS array buffers, *outside* the WebAssembly
// linear memory - so the 460 MB provider does not spend any of the 4 GB a
// wasm32 module has to live in.  What does spend it is the guest's own pages.
console.error('   wasm heap ' + mb(mod.HEAPU8.length) +
              ', node rss ' + mb(process.memoryUsage().rss));

mod.ccall('emu_set_sysroot', null, ['string'], [SYSROOT]);
const argv = [
    '/opt/vv/cudavvm',
    '/opt/vv/libvoicevox_onnxruntime.so.1.17.3',
    '/opt/vv/open_jtalk_dic_utf_8-1.11',
    '/opt/vv/0.vvm',
    '3',
    '@/opt/vv/text.txt',
    '/opt/vv/out.wav',
].join('\0');
const buf = mod._malloc(argv.length + 1);
mod.HEAPU8.set(new TextEncoder().encode(argv + '\0'), buf);

// The program is named by its path in the emscripten filesystem - which is the
// sysroot-prefixed one - while everything in argv is a guest path.  The two
// look alike and are not the same, and getting it wrong reads as "cannot open".
console.error('== running');
const started = Date.now();
const code = mod.ccall('emu_run_path', 'number',
    ['string', 'number', 'number', 'number', 'number'],
    [SYSROOT + '/opt/vv/cudavvm', buf, argv.length, 0, 0]);
const seconds = (Date.now() - started) / 1000;
mod._free(buf);

const insns = mod.ccall('emu_instructions', 'number', [], []);
console.error('');
console.error('exit ' + code + (code < 0 ? ': ' + mod.ccall('emu_error', 'string', [], []) : ''));
console.error(insns.toLocaleString() + ' instructions in ' + seconds.toFixed(1) + ' s'
    + '  (' + (insns / seconds / 1e6).toFixed(1) + ' M/s)');
console.error('wasm heap ' + mb(mod.HEAPU8.length) +
              ', node rss ' + mb(process.memoryUsage().rss));

try {
    const wav = mod.FS.readFile(SYSROOT + '/opt/vv/out.wav');
    const out = path.join(root, 'build/wasm_cuda.wav');
    fs.writeFileSync(out, Buffer.from(wav));
    console.error('wrote ' + out + ', ' + wav.length + ' bytes');
} catch {
    console.error('no out.wav');
}
