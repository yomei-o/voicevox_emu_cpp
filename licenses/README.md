# What is redistributed here, and under what

This repository carries binaries it did not build, so that a clone is runnable
without fetching anything. Each is redistributable, and this is the paperwork.

| in the tree | what | terms |
| --- | --- | --- |
| `guest/0.vvm` | VOICEVOX voice models — 四国めたん, ずんだもん, 春日部つむぎ, 雨晴はう | `voicevox_vvm-TERMS.txt`: 「アプリケーションに組み込んで再配布することができます」. Credit required. |
| `guest/libvoicevox_onnxruntime.so.1.17.3` | VOICEVOX's patched ONNX Runtime | MIT — `voicevox_onnxruntime-TERMS.txt`, `voicevox_onnxruntime-third-party-notices.html` |
| `guest/libvoicevox_core.so` | VOICEVOX CORE | MIT — `voicevox_core-LICENSE.txt` |
| `guest/open_jtalk_dic_utf_8-1.11/` | Open JTalk's system dictionary (NAIST-jdic, compiled for MeCab) | modified BSD — `open_jtalk_dic-COPYING.txt` |
| `guest/predict_duration.onnx` | a dummy model from `VOICEVOX/voicevox_core`, used as the plain-ONNX control | MIT, with the CORE repository |
| `sysroot/lib`, `sysroot/usr/lib`, `sysroot/usr/include` | Debian bookworm glibc, libstdc++ and libgcc | LGPL-2.1+ / GPL-3 with the GCC Runtime Library Exception — `debian-libc6-copyright.txt`, `debian-gcc-12-base-copyright.txt` |

The Debian binaries are unmodified packages; their sources are at
<http://deb.debian.org/debian/pool/main/g/glibc/> and
<http://deb.debian.org/debian/pool/main/g/gcc-12/>, at the exact versions named
in `make_sysroot.sh` (`2.36-9+deb12u14`, `12.2.0-14+deb12u1`), which is also
the script that reproduces this tree from them.

## Credit

Audio produced with these models must carry a credit. For the characters in
`0.vvm` that is one of:

    VOICEVOX:四国めたん
    VOICEVOX:ずんだもん
    VOICEVOX:春日部つむぎ
    VOICEVOX:雨晴はう

Per-character terms: <https://zunko.jp/con_ongen_kiyaku.html> and the links in
`voicevox_vvm-README.txt`.

## The prohibition that does apply

The voice-model terms forbid 「逆コンパイル・リバースエンジニアリング及びこれらの
方法の公開すること」. Running the runtime is its intended use and is not that.
Reading a decrypted model back out of the emulator's guest memory would be, and
this project does not do it.
