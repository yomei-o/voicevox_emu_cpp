// The kernel launches only, in WebAssembly, in seconds rather than a quarter
// of an hour.
//
//   node web/test_cuda_probe.mjs
//
// The full run takes fourteen minutes to build its sessions before it launches
// a single kernel, which is a poor loop to debug in.  src/cudaprobe.c loads one
// small plain .onnx and runs it - eight launches, the same path through the
// shim, seconds instead.
//
// Built with --profiling-funcs and -sASSERTIONS the wasm stack has names in it,
// which "table index is out of bounds" at wasm-function[627] does not.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');
const cuda = process.env.VVCUDA ||
    path.join(process.env.HOME || '', 'vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib');
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
]) put(guest, path.join(root, host));

putTree('/lib/x86_64-linux-gnu', path.join(root, 'guest/cudaguest'));
put('/opt/vv/cudaprobe', path.join(root, 'guest/cudaprobe'));
put('/opt/vv/predict_duration.onnx', path.join(root, 'guest/predict_duration.onnx'));
put('/opt/vv/libvoicevox_onnxruntime.so.1.17.3',
    path.join(cuda, 'libvoicevox_onnxruntime.so.1.17.3'));
put('/opt/vv/libvoicevox_onnxruntime_providers_shared.so',
    path.join(cuda, 'libvoicevox_onnxruntime_providers_shared.so'));
const slim = path.join(cuda, 'libvoicevox_onnxruntime_providers_cuda_slim.so');
put('/opt/vv/libvoicevox_onnxruntime_providers_cuda.so',
    fs.existsSync(slim) ? slim : path.join(cuda, 'libvoicevox_onnxruntime_providers_cuda.so'));

mod.ccall('emu_set_sysroot', null, ['string'], [SYSROOT]);
const argv = [
    '/opt/vv/cudaprobe',
    '/opt/vv/libvoicevox_onnxruntime.so.1.17.3',
    '/opt/vv/predict_duration.onnx',
].join('\0');
const buf = mod._malloc(argv.length + 1);
mod.HEAPU8.set(new TextEncoder().encode(argv + '\0'), buf);

console.error('== running cudaprobe');
const started = Date.now();
const code = mod.ccall('emu_run_path', 'number',
    ['string', 'number', 'number', 'number', 'number'],
    [SYSROOT + '/opt/vv/cudaprobe', buf, argv.length, 0, 0]);
mod._free(buf);
console.error('exit ' + code + (code < 0 ? ': ' + mod.ccall('emu_error', 'string', [], []) : ''));
console.error(mod.ccall('emu_instructions', 'number', [], []).toLocaleString() +
              ' instructions in ' + ((Date.now() - started) / 1000).toFixed(1) + ' s');
