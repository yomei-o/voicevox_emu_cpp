// A session saved by the native build, resumed in WebAssembly.
//
//   node web/test_resume.mjs [~/vv/session.state]
//
// This is the claim the whole guest-space arena was for, and until now it has
// been a claim.  The state was written by a host where a pointer is eight bytes
// and is being read by one where it is four; if anything in the file were a
// host address, or sized by a host word, this is where it would show.
//
// Node rather than a browser because the answer is a number and a browser adds
// nothing to it - and because a 166 MB file goes into MEMFS from disk here
// without a download.
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
const statePath = process.argv[2] || path.join(home, 'vv/session.state');
const modulePath = process.env.VVMODULE || path.join(here, 'x86emu_cuda.js');

for (const f of [statePath, statePath + '.shim']) {
    if (!fs.existsSync(f)) {
        console.error(`no ${f} - take one with tools/wslresume.sh first`);
        process.exit(1);
    }
}

const createX86EmuCuda = require(modulePath);
const decoder = new TextDecoder('utf-8', { fatal: false });
globalThis.x86emuOutput = (fd, bytes) => process.stdout.write(decoder.decode(bytes, { stream: true }));
globalThis.x86emuLog = (line) => console.error('[emu]', line);

const mod = await createX86EmuCuda();
const SYSROOT = '/sysroot';

for (const name of ['VVSTUB_STATS', 'VVSTUB_TIME', 'X86EMU_PROFILE']) {
    if (process.env[name]) {
        mod.ccall('emu_setenv', null, ['string', 'string'], [name, process.env[name]]);
    }
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

// Which files the state needs, from the state.
//
// Pages that still matched the file they were mapped from were left out of it
// and are read back at load time, so every one of those files has to be here,
// under the path the guest knew it by, byte for byte.  Listing them by hand got
// libvoicevox_core.so wrong - the guest's loader found it under
// /lib/x86_64-linux-gnu, the list said /opt/vvcuda - and 351 missing pages
// became an invalid instruction inside a library, a long way from the cause.
//
// The file says which files it needs.  Reading that beats writing it down
// twice.
function statePaths(buf) {
    if (buf.subarray(0, 8).toString('latin1') !== 'X86EMUST') throw new Error('not a state');
    const need = new Set();
    let at = 16;
    while (at + 12 <= buf.length) {
        const tag = buf.subarray(at, at + 4).toString('latin1');
        const len = Number(buf.readBigUInt64LE(at + 4));
        at += 12;
        if (tag === 'END ') break;
        let p = at;
        const u64 = () => { const v = Number(buf.readBigUInt64LE(p)); p += 8; return v; };
        const str = () => { const n = u64(); const s = buf.subarray(p, p + n).toString('utf8'); p += n; return s; };
        if (tag === 'REGN') {
            for (let i = 0, k = u64(); i < k; i++) {
                u64(); u64();            // base, size
                str();                   // name
                const file = str();
                u64();                   // file_offset
                p += 1;                  // contiguous
                if (file) need.add(file);
            }
        } else if (tag === 'FDS ') {
            for (let i = 0, k = u64(); i < k; i++) {
                p += 4;                  // fd
                const file = str();
                p += 7 + 1 + 2 + 8;      // flags, wide_io, cloexec/closed, offset
                if (file) need.add(file);
            }
        }
        at += len;
    }
    return [...need];
}

const stateBuf = fs.readFileSync(statePath);
let staged = 0, absent = [];
// The program itself is not among them: it is an argument, not something the
// guest mapped or opened, so it is named here the way the command line names it.
for (const guestPath of ['/opt/vvcuda/cudavvm', ...statePaths(stateBuf)]) {
    // <stdout> and its two siblings are the standard streams' stand-in names,
    // not files.
    if (guestPath.startsWith('<')) continue;
    const hostPath = path.join(root, 'sysroot', guestPath.replace(/^\//, ''));
    if (!fs.existsSync(hostPath)) { absent.push(guestPath); continue; }
    put(guestPath, hostPath);
    staged++;
}
console.error(`== staged ${staged} files the state names`);
for (const missing of absent) console.error(`   not on disk: ${missing}`);

// The dictionary is read after the resume rather than mapped, so it is not in
// the state's list - it is in the guest's arguments instead.
putTree('/opt/vvcuda/open_jtalk_dic_utf_8-1.11',
        path.join(root, 'sysroot/opt/vvcuda/open_jtalk_dic_utf_8-1.11'));

const text = process.argv[3] || 'ずんだもんなのだ';
mod.FS.writeFile(SYSROOT + '/opt/vvcuda/text.txt', new TextEncoder().encode(text));

// The state itself, and the shim's half.  Outside the sysroot: the emulator
// opens these as host paths, not as something the guest can see.
console.error(`== loading the state (${(stateBuf.length / 1048576).toFixed(1)} MB) into MEMFS`);
mod.FS.mkdirTree('/state');
mod.FS.writeFile('/state/session', new Uint8Array(stateBuf));
mod.FS.writeFile('/state/session.shim', new Uint8Array(fs.readFileSync(statePath + '.shim')));

mod.ccall('emu_set_sysroot', null, ['string'], [SYSROOT]);
// argv has to be the argv the state was saved with: it was written into the
// guest's stack when it was loaded and came back with the rest of its memory,
// so what is passed here is only what load() needs to get as far as being
// overwritten.  Saying the same thing twice costs nothing and keeps the two
// runs comparable.
const argv = [
    '/opt/vvcuda/cudavvm',
    '/opt/vvcuda/libvoicevox_onnxruntime.so.1.17.3',
    '/opt/vvcuda/open_jtalk_dic_utf_8-1.11',
    '/opt/vvcuda/0.vvm',
    '3',
    '@/opt/vvcuda/text.txt',
    '/opt/vvcuda/out.wav',
].join('\0');
const buf = mod._malloc(argv.length + 1);
mod.HEAPU8.set(new TextEncoder().encode(argv + '\0'), buf);

console.error(`== resuming, and saying: ${text}`);
const started = Date.now();
const code = mod.ccall('emu_resume_path', 'number',
    ['string', 'number', 'number', 'number', 'number', 'string'],
    [SYSROOT + '/opt/vvcuda/cudavvm', buf, argv.length, 0, 0, '/state/session']);
mod._free(buf);
const seconds = (Date.now() - started) / 1000;
console.error('exit ' + code + (code < 0 ? ': ' + mod.ccall('emu_error', 'string', [], []) : ''));
console.error(mod.ccall('emu_instructions', 'number', [], []).toLocaleString() +
              ' instructions in ' + seconds.toFixed(1) + ' s');

if (code >= 0) {
    const wav = SYSROOT + '/opt/vvcuda/out.wav';
    if (mod.FS.analyzePath(wav).exists) {
        const bytes = mod.FS.readFile(wav);
        const out = process.argv[4] || path.join(root, 'build/wasm_resume.wav');
        fs.mkdirSync(path.dirname(out), { recursive: true });
        fs.writeFileSync(out, Buffer.from(bytes));
        console.error(`wrote ${out}, ${bytes.length} bytes`);
    } else {
        console.error('no out.wav - the resumed guest did not get that far');
        process.exit(1);
    }
}
process.exit(code < 0 ? 1 : 0);
