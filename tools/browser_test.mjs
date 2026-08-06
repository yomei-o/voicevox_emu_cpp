// Drives web/index.html in a real browser and checks that the demo works.
//
// The page rehearsal in web/test_page.mjs covers everything but the DOM; this
// covers the DOM.  It serves the repository the way GitHub Pages will, launches
// Chrome (or Edge) headless, clicks the buttons, and fails on a console error
// or a result that does not parse - which is the only way to find out that a
// button is wired to the wrong id without opening a tab.
//
//   node tools/browser_test.mjs
//   node tools/browser_test.mjs --speak     # also runs synthesis: hours
//   node tools/browser_test.mjs --url https://yomei-o.github.io/voicevox_emu_cpp/web/
//   BROWSER="C:/path/to/msedge.exe" node tools/browser_test.mjs
//
// With --url it tests the deployed site instead of the working tree, which is
// the only way to find out that Pages is serving a payload file as something
// the page cannot use.
//
// Uses the DevTools protocol directly over node's built-in WebSocket, so there
// is nothing to install.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');
const alsoSpeak = process.argv.includes('--speak');
const urlArg = process.argv.indexOf('--url');
const liveUrl = urlArg >= 0 ? process.argv[urlArg + 1] : null;

// Whichever of these actually opens a debugging port wins.  Not every install
// will: a managed machine can forbid remote debugging by policy, and Chrome
// says so and exits while Edge beside it is happy to oblige.
const CANDIDATES = [
    process.env.BROWSER,
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/microsoft-edge',
].filter(Boolean).filter((p) => fs.existsSync(p));
if (!CANDIDATES.length) {
    console.error('no chrome or edge found; set BROWSER=');
    process.exit(2);
}

// ---- serve the repository -------------------------------------------------

const types = {'.json': 'application/json', '.html': 'text/html', '.js': 'text/javascript'};
const server = createServer((req, res) => {
    let rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
    if (rel === '' || rel.endsWith('/')) rel += 'index.html';
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
let url = liveUrl;
if (!liveUrl) {
    await new Promise((r) => server.listen(0, '127.0.0.1', r));
    url = `http://127.0.0.1:${server.address().port}/web/index.html`;
    console.log(`serving ${root}`);
} else {
    server.close();
}
console.log(`page     ${url}`);

// ---- launch it ------------------------------------------------------------

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'vvdemo-'));
const port = 9222 + (process.pid % 300);

async function targetOn(portNo, tries) {
    for (let i = 0; i < tries; i++) {
        try {
            const list = await (await fetch(`http://127.0.0.1:${portNo}/json/list`)).json();
            const page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
            if (page) return page.webSocketDebuggerUrl;
        } catch (e) { /* not up yet */ }
        await new Promise((r) => setTimeout(r, 250));
    }
    return null;
}

let child = null;
let target = null;
let browser = null;
for (const candidate of CANDIDATES) {
    child = spawn(candidate, [
        '--headless=new',
        '--disable-gpu',
        '--no-first-run',
        '--no-default-browser-check',
        `--user-data-dir=${profile}`,
        `--remote-debugging-port=${port}`,
        url,
    ], {stdio: ['ignore', 'ignore', 'pipe']});
    child.stderr.on('data', () => {});  // they are all chatty on stderr
    target = await targetOn(port, 60);
    if (target) {
        browser = candidate;
        break;
    }
    child.kill();
}
if (!target) {
    console.error('no browser opened a debugging port (a policy may forbid it)');
    process.exit(2);
}
console.log(`browser  ${browser}`);

const ws = new WebSocket(target);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });

let nextId = 1;
const pending = new Map();
const consoleErrors = [];
ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.id && pending.has(msg.id)) {
        const {resolve, reject} = pending.get(msg.id);
        pending.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
        return;
    }
    if (msg.method === 'Runtime.exceptionThrown')
        consoleErrors.push(msg.params.exceptionDetails.exception?.description ||
                           msg.params.exceptionDetails.text);
    if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error')
        consoleErrors.push(msg.params.args.map((a) => a.value ?? a.description).join(' '));
};

const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, {resolve, reject});
    ws.send(JSON.stringify({id, method, params}));
});

await send('Runtime.enable');
await send('Page.enable');

async function evaluate(expression) {
    const r = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (r.exceptionDetails)
        throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
}

// The page may still be loading when the debugger attaches.
async function waitFor(expression, timeoutMs, what) {
    const until = Date.now() + timeoutMs;
    for (;;) {
        if (await evaluate(expression)) return;
        if (Date.now() > until) throw new Error(`timed out waiting for ${what}`);
        await new Promise((r) => setTimeout(r, 500));
    }
}

let failures = 0;
function check(ok, what) {
    console.log(`${ok ? 'ok   ' : 'FAIL '} ${what}`);
    if (!ok) failures++;
}

try {
    await waitFor(`!!document.getElementById('prep-analyze')`, 30000, 'the page to load');
    check(true, 'the page loads');

    console.log('     downloading the analysis payload (33 MB)...');
    await evaluate(`document.getElementById('prep-analyze').click(), true`);
    await waitFor(`!document.getElementById('go-analyze').disabled`, 600000,
                  'the analysis payload');
    check(true, 'the analysis payload loads and the emulator starts');

    console.log('     running text analysis...');
    await evaluate(`document.getElementById('text-analyze').value = 'ずんだもんなのだ', true`);
    await evaluate(`document.getElementById('go-analyze').click(), true`);
    await waitFor(`document.getElementById('analyze-out').textContent.length > 100`, 300000,
                  'the analysis result');

    const out = await evaluate(`document.getElementById('analyze-out').textContent`);
    const took = await evaluate(`document.getElementById('analyze-time').textContent`);
    let phrases = null;
    try {
        phrases = JSON.parse(out);
    } catch (e) { /* reported below */ }
    check(Array.isArray(phrases) && phrases.length > 0, 'the result is accent phrase JSON');
    if (Array.isArray(phrases)) {
        const moras = phrases.reduce((n, p) => n + (p.moras?.length || 0), 0);
        console.log(`     ${phrases.length} accent phrases, ${moras} moras, ${took}`);
        check(moras === 8, 'ずんだもんなのだ comes out as 8 moras');
        check(phrases[0].moras[0].text === 'ズ', 'the first mora is ズ');
    }

    if (alsoSpeak) {
        console.log('     downloading the synthesis payload (76 MB)...');
        await evaluate(`document.getElementById('prep-speak').click(), true`);
        await waitFor(`!document.getElementById('go-speak').disabled`, 1800000,
                      'the synthesis payload');
        check(true, 'the synthesis payload loads');
        console.log('     synthesising - this takes hours');
        await evaluate(`document.getElementById('text-speak').value = 'あ', true`);
        await evaluate(`document.getElementById('go-speak').click(), true`);
        await waitFor(`!!document.querySelector('#player audio')`, 8 * 3600 * 1000,
                      'the audio');
        check(true, 'synthesis produces a WAV the page can play');
        // Pull the bytes back out so the run leaves evidence rather than a
        // claim: the page holds them in a blob: URL its own audio element uses.
        const b64 = await evaluate(`(async () => {
            const src = document.querySelector('#player audio').src;
            const buf = await (await fetch(src)).arrayBuffer();
            let s = '';
            const b = new Uint8Array(buf);
            for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]);
            return btoa(s);
        })()`);
        const wav = Buffer.from(b64, 'base64');
        fs.writeFileSync(path.join(root, 'browser_out.wav'), wav);
        console.log(`     wrote browser_out.wav, ${wav.length} bytes`);
        check(wav.length > 44 && wav.slice(0, 4).toString() === 'RIFF',
              'the WAV is a real RIFF file');
        const took = await evaluate(`document.querySelector('#player .note').textContent`);
        console.log(`     ${took.replace(/\s+/g, ' ').trim()}`);
    }

    check(consoleErrors.length === 0, 'no console errors');
    for (const e of consoleErrors) console.log(`     ${e}`);
} catch (err) {
    console.log(`FAIL  ${err.message}`);
    for (const e of consoleErrors) console.log(`     ${e}`);
    failures++;
}

ws.close();
child.kill();
if (!liveUrl) server.close();
try {
    fs.rmSync(profile, {recursive: true, force: true});
} catch (e) { /* windows sometimes holds it briefly */ }

console.log(failures ? `\n${failures} failed` : '\nBROWSER TEST OK');
process.exit(failures ? 1 : 0);
