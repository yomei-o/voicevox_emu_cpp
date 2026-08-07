// The CUDA runtime the repository now carries: does it load, and is it the
// same runtime?
//
//   node web/test_carried.mjs
//
// The demo page used to refuse to start until three files had been picked out
// of a 440 MB download.  They are here now, gzipped, and the page fetches them
// like everything else - so what has to be checked is that the gzipped, slimmed
// copies are still a working ONNX Runtime.  cudaprobe is the cheap way to ask:
// eight kernel launches and sixteen reference values, in seconds rather than
// the twenty minutes a session build costs.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');
const modulePath = process.env.VVMODULE || path.join(here, 'x86emu_cuda.js');

const createX86EmuCuda = require(modulePath);
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
function putGz(guestPath, hostPath) {
    const full = SYSROOT + guestPath;
    mod.FS.mkdirTree(path.posix.dirname(full));
    mod.FS.writeFile(full, new Uint8Array(gunzipSync(fs.readFileSync(hostPath))));
}
function putTree(guestDir, hostDir) {
    for (const name of fs.readdirSync(hostDir)) {
        const h = path.join(hostDir, name);
        if (fs.statSync(h).isDirectory()) putTree(guestDir + '/' + name, h);
        else put(guestDir + '/' + name, h);
    }
}

for (const [guest, host] of [
    ['/lib64/ld-linux-x86-64.so.2', 'sysroot/lib64/ld-linux-x86-64.so.2'],
    ['/lib/x86_64-linux-gnu/libc.so.6', 'sysroot/lib/x86_64-linux-gnu/libc.so.6'],
    ['/lib/x86_64-linux-gnu/libm.so.6', 'sysroot/lib/x86_64-linux-gnu/libm.so.6'],
    ['/lib/x86_64-linux-gnu/libdl.so.2', 'sysroot/lib/x86_64-linux-gnu/libdl.so.2'],
    ['/lib/x86_64-linux-gnu/libpthread.so.0', 'sysroot/lib/x86_64-linux-gnu/libpthread.so.0'],
    ['/lib/x86_64-linux-gnu/libgcc_s.so.1', 'sysroot/lib/x86_64-linux-gnu/libgcc_s.so.1'],
    ['/usr/lib/x86_64-linux-gnu/libstdc++.so.6', 'sysroot/usr/lib/x86_64-linux-gnu/libstdc++.so.6'],
    ['/proc/cpuinfo', 'sysroot/proc/cpuinfo'],
    ['/opt/vv/cudaprobe', 'guest/cudaprobe'],
    ['/opt/vv/predict_duration.onnx', 'guest/predict_duration.onnx'],
]) put(guest, path.join(root, host));
putTree('/lib/x86_64-linux-gnu', path.join(root, 'guest/cudaguest'));

// The point of this test: these come from the repository, gzipped, and nothing
// else is supplied.
for (const [guest, host] of [
    ['/opt/vv/libvoicevox_onnxruntime.so.1.17.3',
     'guest/cuda/libvoicevox_onnxruntime.so.1.17.3.gz'],
    ['/opt/vv/libvoicevox_onnxruntime_providers_shared.so',
     'guest/cuda/libvoicevox_onnxruntime_providers_shared.so.gz'],
    ['/opt/vv/libvoicevox_onnxruntime_providers_cuda.so',
     'guest/cuda/libvoicevox_onnxruntime_providers_cuda.so.gz'],
]) putGz(guest, path.join(root, host));

mod.ccall('emu_set_sysroot', null, ['string'], [SYSROOT]);
const argv = [
    '/opt/vv/cudaprobe',
    '/opt/vv/libvoicevox_onnxruntime.so.1.17.3',
    '/opt/vv/predict_duration.onnx',
].join('\0');
const buf = mod._malloc(argv.length + 1);
mod.HEAPU8.set(new TextEncoder().encode(argv + '\0'), buf);
const code = mod.ccall('emu_run_path', 'number',
    ['string', 'number', 'number', 'number', 'number'],
    [SYSROOT + '/opt/vv/cudaprobe', buf, argv.length, 0, 0]);
mod._free(buf);
console.error('exit ' + code + (code < 0 ? ': ' + mod.ccall('emu_error', 'string', [], []) : ''));
process.exit(code < 0 ? 1 : 0);
