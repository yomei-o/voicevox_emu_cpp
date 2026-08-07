// What web/cuda.html does, without a browser: build a session, save it, resume
// from it, and say two different things.
//
//   node web/test_roundtrip.mjs
//
// The page's own path is the one worth checking, and checking it by hand means
// twenty minutes of watching a progress bar.  This is the same sequence through
// the same entry points, so a break in it is a break in the page.
//
// It is slow on purpose - the whole point is that the first step is slow and
// the rest are not.
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');
const home = process.env.HOME || '';
const cuda = process.env.VVCUDA ||
    path.join(home, 'vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib');
const modulePath = process.env.VVMODULE || path.join(here, 'x86emu_cuda.js');

const createX86EmuCuda = require(modulePath);
const decoder = new TextDecoder('utf-8', { fatal: false });
globalThis.x86emuOutput = (fd, bytes) => process.stdout.write(decoder.decode(bytes, { stream: true }));
globalThis.x86emuLog = (line) => console.error('[emu]', line);

const mod = await createX86EmuCuda();
const SYSROOT = '/sysroot';
const STATE = '/state/session';

for (const name of ['VVSTUB_STATS', 'VVSTUB_TIME']) {
    if (process.env[name]) mod.ccall('emu_setenv', null, ['string', 'string'], [name, process.env[name]]);
}

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

// The same guest paths web/cuda_worker.js uses, which are the same ones
// tools/wslrun_cuda.sh uses.  That is what makes a session built by one of them
// resume in the others.
for (const [guest, host] of [
    ['/lib64/ld-linux-x86-64.so.2', 'sysroot/lib64/ld-linux-x86-64.so.2'],
    ['/lib/x86_64-linux-gnu/libc.so.6', 'sysroot/lib/x86_64-linux-gnu/libc.so.6'],
    ['/lib/x86_64-linux-gnu/libm.so.6', 'sysroot/lib/x86_64-linux-gnu/libm.so.6'],
    ['/lib/x86_64-linux-gnu/libdl.so.2', 'sysroot/lib/x86_64-linux-gnu/libdl.so.2'],
    ['/lib/x86_64-linux-gnu/libpthread.so.0', 'sysroot/lib/x86_64-linux-gnu/libpthread.so.0'],
    ['/lib/x86_64-linux-gnu/libgcc_s.so.1', 'sysroot/lib/x86_64-linux-gnu/libgcc_s.so.1'],
    ['/usr/lib/x86_64-linux-gnu/libstdc++.so.6', 'sysroot/usr/lib/x86_64-linux-gnu/libstdc++.so.6'],
    ['/proc/cpuinfo', 'sysroot/proc/cpuinfo'],
    ['/lib/x86_64-linux-gnu/libvoicevox_core.so', 'guest/libvoicevox_core.so'],
    ['/opt/vvcuda/cudavvm', 'guest/cudavvm'],
    ['/opt/vvcuda/0.vvm', 'guest/0.vvm'],
]) put(guest, path.join(root, host));
putTree('/lib/x86_64-linux-gnu', path.join(root, 'guest/cudaguest'));
put('/opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3',
    path.join(cuda, 'libvoicevox_onnxruntime.so.1.17.3'));
put('/opt/vvcuda/libvoicevox_onnxruntime_providers_shared.so',
    path.join(cuda, 'libvoicevox_onnxruntime_providers_shared.so'));
const slim = path.join(home, 'vv/slimtest/libvoicevox_onnxruntime_providers_cuda.so');
put('/opt/vvcuda/libvoicevox_onnxruntime_providers_cuda.so',
    fs.existsSync(slim) ? slim : path.join(cuda, 'libvoicevox_onnxruntime_providers_cuda.so'));
// The dictionary the way the page gets it - out of the tarball, into the
// parent, because every entry in it is already prefixed with the directory's
// name.  Copying an unpacked tree instead is easier and tests a path the page
// does not take: it hid a bug where the page named the directory twice and
// Mecab_load could not find sys.dic.
{
    const { execFileSync } = await import('node:child_process');
    const dir = path.join(root, 'build/dic');
    fs.rmSync(dir, { recursive: true, force: true });
    fs.mkdirSync(dir, { recursive: true });
    execFileSync('tar', ['xzf', path.join(root, 'guest/open_jtalk_dic_utf_8-1.11.tar.gz'),
                         '-C', dir]);
    for (const name of fs.readdirSync(dir)) putTree('/opt/vvcuda/' + name, path.join(dir, name));
}

mod.ccall('emu_set_sysroot', null, ['string'], [SYSROOT]);
mod.FS.mkdirTree('/state');

const ARGS = [
    '/opt/vvcuda/cudavvm',
    '/opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3',
    '/opt/vvcuda/open_jtalk_dic_utf_8-1.11',
    '/opt/vvcuda/0.vvm',
    '3',
    '@/opt/vvcuda/text.txt',
    '/opt/vvcuda/out.wav',
].join('\0');

function run(text, state) {
    mod.FS.writeFile(SYSROOT + '/opt/vvcuda/text.txt', new TextEncoder().encode(text));
    const buf = mod._malloc(ARGS.length + 1);
    mod.HEAPU8.set(new TextEncoder().encode(ARGS + '\0'), buf);
    const started = Date.now();
    const code = mod.ccall('emu_resume_path', 'number',
        ['string', 'number', 'number', 'number', 'number', 'string'],
        [SYSROOT + '/opt/vvcuda/cudavvm', buf, ARGS.length, 0, 0, state || '']);
    mod._free(buf);
    const seconds = (Date.now() - started) / 1000;
    if (code !== 0) {
        console.error('exit ' + code + ': ' + mod.ccall('emu_error', 'string', [], []));
        process.exit(1);
    }
    const insns = mod.ccall('emu_instructions', 'number', [], []);
    console.error(`  ${seconds.toFixed(1)} s, ${(insns / 1e6).toFixed(1)} M instructions, ` +
                  `heap ${(mod.HEAPU8.length / 1048576).toFixed(0)} MB`);
    return seconds;
}

console.error('== building the sessions and saving (the slow one)');
// The guest's environment, not this process's.  emu_setenv is the other one
// and the guest cannot see it - setting VVSNAPSHOT there ran the whole thing
// through to synthesis without ever taking a snapshot.
mod.ccall('emu_guest_setenv', null, ['string', 'string'], ['VVSNAPSHOT', STATE]);
run('あ', null);
mod.ccall('emu_guest_setenv', null, ['string', 'string'], ['VVSNAPSHOT', '']);
const size = mod.FS.stat(STATE).size;
const shimSize = mod.FS.stat(STATE + '.shim').size;
console.error(`  session.state ${(size / 1048576).toFixed(1)} MB, .shim ${shimSize} bytes`);

fs.mkdirSync(path.join(root, 'build'), { recursive: true });
for (const [phrase, name] of [['ずんだもんなのだ', 'a'], ['こんにちは、げんきですか', 'b']]) {
    console.error(`== resuming and saying: ${phrase}`);
    run(phrase, STATE);
    const wav = mod.FS.readFile(SYSROOT + '/opt/vvcuda/out.wav');
    const out = path.join(root, `build/roundtrip_${name}.wav`);
    fs.writeFileSync(out, Buffer.from(wav));
    console.error(`  wrote ${out}, ${wav.length} bytes`);
}
