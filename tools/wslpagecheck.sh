#!/bin/sh
# Does the demo page's own path work?
#
# Not the browser - there is none here - but everything the page does that can
# fail without one: that the module loads under importScripts, that every URL
# the worker fetches exists, and that the three CUDA libraries are told apart by
# name the way the file picker will hand them over.
set -e
cd "$(dirname "$0")/.."
NODE=$(ls "$HOME"/emsdk/node/*/bin/node | head -1)
CUDA=${VVCUDA:-$HOME/vv/cuda/voicevox_onnxruntime-linux-x64-cuda-1.17.3/lib}

echo "== every URL the worker fetches, from web/"
missing=0
for url in $(sed -n "s/.*'\(\.\.\/[^']*\|guest\/[^']*\)'.*/\1/p" web/cuda_worker.js | sort -u); do
    if [ -f "web/$url" ]; then
        printf '  ok      %s\n' "$url"
    else
        printf '  MISSING %s\n' "$url"
        missing=$((missing + 1))
    fi
done
[ "$missing" = 0 ] || { echo "$missing missing"; exit 1; }

echo
echo "== the module loads and exports what the worker calls"
"$NODE" -e "
const create = require('./web/x86emu_cuda.js');
create().then(m => {
    for (const name of ['emu_run_path','emu_set_sysroot','emu_setenv','emu_error','emu_instructions']) {
        if (typeof m['_' + name] !== 'function') { console.log('  MISSING ' + name); process.exit(1); }
        console.log('  ok      ' + name);
    }
    for (const name of ['FS','ccall','cwrap','HEAPU8']) {
        if (!m[name]) { console.log('  MISSING runtime ' + name); process.exit(1); }
        console.log('  ok      ' + name);
    }
});
"

echo
echo "== the file picker tells the three apart"
"$NODE" -e "
const src = require('fs').readFileSync('web/cuda_worker.js', 'utf8');
const body = src.slice(src.indexOf('function classify'));
const classify = new Function('name', body.slice(body.indexOf('{') + 1, body.indexOf('\n}')));
for (const n of ['libvoicevox_onnxruntime.so.1.17.3',
                 'libvoicevox_onnxruntime_providers_shared.so',
                 'libvoicevox_onnxruntime_providers_cuda.so',
                 'libvoicevox_onnxruntime_providers_cuda_slim.so',
                 'something_else.so']) {
    console.log('  ' + (classify(n) || '(ignored)').padEnd(52) + ' <- ' + n);
}
"
