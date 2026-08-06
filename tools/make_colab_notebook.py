# Writes colab/voicevox_cuda_reference.ipynb.
#
# The notebook it makes is the ground truth for the CUDA shim: the same programs
# this repository runs against stand-in CUDA libraries, run on a real GPU
# against the real ones.  Whatever comes out is what the shim has to reproduce.
#
#   python tools/make_colab_notebook.py
#
# Generated rather than hand-written because a .ipynb is JSON with every line of
# source as a separate string, which is not a thing to maintain by hand.
import json
import os

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")}

def code(text):
    lines = text.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
    }

REPO = "https://raw.githubusercontent.com/yomei-o/voicevox_emu_cpp/main"

cells = [
md("""
# VOICEVOX CORE on a real GPU — the reference for the CUDA shim

`voicevox_emu_cpp` runs the **CUDA** build of `voicevox_onnxruntime` with no GPU at
all, against stand-in `libcudart` / `cuBLAS` / `cuDNN` that do the arithmetic on
the CPU. The question this notebook answers is the only one that matters about
that: **does it get the same numbers as real CUDA?**

So this runs the same two programs — `cudaprobe` and `cudavvm`, fetched from the
repository unchanged — on a real GPU against the real libraries, and prints the
numbers and a WAV. Compare those against the local run.

**Runtime → Change runtime type → GPU** before running.
"""),

code("""
!nvidia-smi -L || echo 'no GPU: Runtime -> Change runtime type -> GPU'
!nvcc --version | tail -2
"""),

md("""
## 1. The pieces

The same versions the repository pins: `voicevox_onnxruntime` 1.17.3 (the CUDA
build, 103 MB), CORE 0.16.4, ずんだもん's voice model, and the Open JTalk
dictionary.
"""),

code("""
import os, subprocess, textwrap
os.makedirs('/content/vv', exist_ok=True)
os.chdir('/content/vv')

VV_ORT   = '1.17.3'
VV_CORE  = '0.16.4'
VV_VVM   = '0.16.4'   # the vvm repository tags with CORE's version
VVM      = '0.vvm'

def get(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f'have     {out}  ({os.path.getsize(out):,} bytes)'); return
    print(f'fetching {out}')
    # -f so a 404 is a failure rather than an HTML page saved as a .tgz, and the
    # status is printed either way: a truncated download is the failure mode
    # that looks like a corrupt archive three cells later.
    r = subprocess.run(['curl', '-sfL', '-w', '%{http_code}', '-o', out, url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f'download failed: {url} '
                         f'(curl exit {r.returncode}, http {r.stdout.strip() or "?"})')
    print(f'         {out}  ({os.path.getsize(out):,} bytes, http {r.stdout.strip()})')

get(f'https://github.com/VOICEVOX/onnxruntime-builder/releases/download/voicevox_onnxruntime-{VV_ORT}/voicevox_onnxruntime-linux-x64-cuda-{VV_ORT}.tgz', 'ort.tgz')
get(f'https://github.com/VOICEVOX/voicevox_core/releases/download/{VV_CORE}/voicevox_core-linux-x64-{VV_CORE}.zip', 'core.zip')
get(f'https://github.com/VOICEVOX/voicevox_vvm/releases/download/{VV_VVM}/{VVM}', VVM)
# The dictionary from this repository rather than from SourceForge: SourceForge
# redirects to a mirror, and from Colab that mirror is often unreachable
# (curl exit 7 behind an http 302).  It is the same archive - see
# licenses/README.md - and it is already here.
get('https://raw.githubusercontent.com/yomei-o/voicevox_emu_cpp/main/guest/open_jtalk_dic_utf_8-1.11.tar.gz', 'ojdic.tar.gz')

!tar xzf ort.tgz && unzip -oq core.zip && tar xzf ojdic.tar.gz
!cp voicevox_onnxruntime-linux-x64-cuda-{VV_ORT}/lib/*.so* .
!cp voicevox_core-linux-x64-{VV_CORE}/lib/libvoicevox_core.so .
!cp voicevox_core-linux-x64-{VV_CORE}/include/voicevox_core.h .

# A short dictionary means the download was cut off, and that only shows up
# much later as an unhelpful failure inside Open JTalk.
import os
assert os.path.exists('open_jtalk_dic_utf_8-1.11/sys.dic'), 'dictionary incomplete'
print('sys.dic', f"{os.path.getsize('open_jtalk_dic_utf_8-1.11/sys.dic'):,}", 'bytes')
!ls -la *.so* {VVM} | head
"""),

md("""
### The CUDA libraries the provider actually wants

`libvoicevox_onnxruntime_providers_cuda.so` is linked against **cuDNN 8**, and a
current Colab image ships cuDNN 9 - so loading it fails with
`libcudnn.so.8: cannot open shared object file`. NVIDIA publishes the 8 series
on PyPI, which needs no root and lands in site-packages.
"""),

code("""
import glob, os, subprocess, sys

def find(soname):
    hits = []
    for root in ['/usr/local', '/usr/lib', '/opt',
                 os.path.dirname(os.path.dirname(sys.executable)) + '/lib']:
        hits += glob.glob(f'{root}/**/{soname}', recursive=True)
    return sorted(set(os.path.dirname(h) for h in hits))

if not find('libcudnn.so.8'):
    print('installing cuDNN 8 for CUDA 12')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'nvidia-cudnn-cu12==8.9.7.29'], check=True)

dirs = []
for so in ['libcudnn.so.8', 'libcublas.so.12', 'libcublasLt.so.12',
           'libcufft.so.11', 'libcudart.so.12']:
    d = find(so)
    print(f'{so:22} {d[0] if d else "NOT FOUND"}')
    dirs += d

# Everything the provider needs, ahead of whatever else is on the path.
LD = ':'.join(dict.fromkeys(dirs + ['/content/vv', '/usr/local/cuda/lib64']))
os.environ['LD_LIBRARY_PATH'] = LD
print()
print('LD_LIBRARY_PATH =', LD)
"""),

md("""
## 2. A plain `.onnx` to start with

`predict_duration.onnx` is the small model VOICEVOX ships for testing — no
encryption, four convolutions, and the four kernels the shim implements first.
Two runs: the CPU provider, which is the arithmetic everyone agrees on, and the
CUDA provider on the real GPU.
"""),

code("""
REPO = '""" + REPO + """'
for f in ['src/cudaprobe.c', 'src/cudavvm.c', 'src/onnxruntime_c_api.h']:
    subprocess.run(['curl', '-sfL', '-o', os.path.basename(f), f'{REPO}/{f}'], check=True)
subprocess.run(['curl', '-sfL', '-o', 'predict_duration.onnx',
                f'{REPO}/guest/predict_duration.onnx'], check=True)
!gcc -O2 -Wall -I. -o cudaprobe cudaprobe.c -ldl
!gcc -O2 -Wall -I. -o cudavvm cudavvm.c -L. -lvoicevox_core -Wl,-rpath,'$ORIGIN'
!ls -la cudaprobe cudavvm
"""),

code("""
print('================ CPU provider (the arithmetic everyone agrees on)')
!LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./cudaprobe ./libvoicevox_onnxruntime.so.{VV_ORT} ./predict_duration.onnx --cpu 2>/dev/null | tail -20
"""),

code("""
print('================ CUDA provider, on the real GPU')
!LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./cudaprobe ./libvoicevox_onnxruntime.so.{VV_ORT} ./predict_duration.onnx 2>/dev/null | tail -20
"""),

md("""
The two should agree to within float rounding. If they do, the CUDA path is
sound and the shim has an unambiguous target.
"""),

md("""
## 3. The whole pipeline, on the GPU

`cudavvm` is the same program the repository runs against the stand-ins: load
the runtime, build a synthesizer with `acceleration_mode = GPU`, decrypt the
voice model, and speak. Here it does it for real.
"""),

code("""
print('================ the full pipeline on the GPU')
!LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./cudavvm ./libvoicevox_onnxruntime.so.{VV_ORT} ./open_jtalk_dic_utf_8-1.11 ./{VVM} 3 2>&1 | grep -v '^kernel ' | tail -25
"""),

md("""
## 4. The reference WAV

`cudavvm` throws its audio away — it was written to enumerate kernels, not to
listen. This produces one to compare against, with the same text and style, and
a checksum so two runs can be compared without ears.
"""),

code("""
say_c = r'''
#include <stdio.h>
#include <stdlib.h>
#include "voicevox_core.h"
int main(int argc, char** argv) {
    const char* ort = argv[1]; const char* dict = argv[2];
    const char* vvm = argv[3]; const char* text = argv[4];
    uint32_t style = (uint32_t)atoi(argv[5]); const char* out = argv[6];
    VoicevoxLoadOnnxruntimeOptions o = voicevox_make_default_load_onnxruntime_options();
    o.filename = ort;
    const VoicevoxOnnxruntime* rt = NULL;
    if (voicevox_onnxruntime_load_once(o, &rt)) { puts("load_once failed"); return 1; }
    OpenJtalkRc* ojt = NULL;
    if (voicevox_open_jtalk_rc_new(dict, &ojt)) { puts("open_jtalk failed"); return 1; }
    VoicevoxInitializeOptions io = voicevox_make_default_initialize_options();
    io.acceleration_mode = VOICEVOX_ACCELERATION_MODE_GPU;
    VoicevoxSynthesizer* syn = NULL;
    if (voicevox_synthesizer_new(rt, ojt, io, &syn)) { puts("synthesizer_new failed"); return 1; }
    printf("gpu_mode=%d\\n", (int)voicevox_synthesizer_is_gpu_mode(syn));
    VoicevoxVoiceModelFile* m = NULL;
    if (voicevox_voice_model_file_open(vvm, &m)) { puts("model open failed"); return 1; }
    if (voicevox_synthesizer_load_voice_model(syn, m)) { puts("load_voice_model failed"); return 1; }
    VoicevoxTtsOptions t = voicevox_make_default_tts_options();
    uintptr_t len = 0; uint8_t* wav = NULL;
    VoicevoxResultCode r = voicevox_synthesizer_tts(syn, text, style, t, &len, &wav);
    if (r) { printf("tts failed %d\\n", (int)r); return 1; }
    FILE* f = fopen(out, "wb"); fwrite(wav, 1, len, f); fclose(f);
    printf("wrote %s %zu bytes\\n", out, (size_t)len);
    return 0;
}
'''
open('gpusay.c', 'w').write(say_c)
!gcc -O2 -Wall -I. -o gpusay gpusay.c -L. -lvoicevox_core -Wl,-rpath,'$ORIGIN'
!LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./gpusay ./libvoicevox_onnxruntime.so.{VV_ORT} ./open_jtalk_dic_utf_8-1.11 ./{VVM} 'あ' 3 gpu_a.wav 2>&1 | tail -3
!LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./gpusay ./libvoicevox_onnxruntime.so.{VV_ORT} ./open_jtalk_dic_utf_8-1.11 ./{VVM} 'ずんだもんなのだ' 3 gpu_zundamon.wav 2>&1 | tail -3
"""),

code("""
import hashlib, struct
for name in ['gpu_a.wav', 'gpu_zundamon.wav']:
    b = open(name, 'rb').read()
    n = (len(b) - 44) // 2
    s = struct.unpack('<%dh' % n, b[44:44 + n * 2])
    print(f'{name}: {len(b)} bytes, {n} samples, peak {max(abs(v) for v in s)}, '
          f'sha256 {hashlib.sha256(b).hexdigest()[:16]}')
"""),

code("""
from IPython.display import Audio, display
for name in ['gpu_a.wav', 'gpu_zundamon.wav']:
    print(name)
    display(Audio(name))
"""),

md("""
## 5. Taking the numbers home

Download these and compare against the local run:

- `gpu_a.wav`, `gpu_zundamon.wav` — the audio a real GPU produces
- the printed `predict_duration` values, which are the first thing a shim has
  to match

`tools/wavcmp.mjs` in the repository does the comparison:

    node tools/wavcmp.mjs gpu_a.wav shim_a.wav

A difference of a few least-significant bits is the same speech; anything you
could hear is percent, not thousandths.

**Credit.** Audio made with these models carries `VOICEVOX:ずんだもん`.
"""),

code("""
from google.colab import files
for f in ['gpu_a.wav', 'gpu_zundamon.wav']:
    files.download(f)
"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

os.makedirs("colab", exist_ok=True)
path = os.path.join("colab", "voicevox_cuda_reference.ipynb")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", path, os.path.getsize(path), "bytes,", len(cells), "cells")
