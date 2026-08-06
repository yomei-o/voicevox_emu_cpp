// How different are two WAVs, really?
//
// The emulator does not reproduce a real CPU bit for bit and is not trying to:
// RSQRTPS and RCPPS are *approximate* by definition, hardware answers them to
// about twelve bits, and this emulator computes them exactly - which is inside
// what the architecture allows and outside what a byte comparison forgives.
// Native ONNX Runtime also picks AVX2 kernels where the emulated one is told it
// has only SSE2, and the two sum in different orders.
//
// So "identical" is the wrong question and this asks the right one: how far
// apart are the samples, against the peak of the signal.
//
//   node tools/wavcmp.mjs a.wav b.wav
//
// A difference of a few least-significant bits on a 16-bit signal is the same
// speech; anything you could hear would be percent, not thousandths.
import fs from 'node:fs';

const [, , pathA, pathB] = process.argv;
if (!pathA || !pathB) {
    console.error('usage: node tools/wavcmp.mjs a.wav b.wav');
    process.exit(2);
}

const a = fs.readFileSync(pathA);
const b = fs.readFileSync(pathB);
const HEADER = 44;  // canonical RIFF/WAVE, which is what CORE writes

if (a.length !== b.length)
    console.log(`lengths differ: ${a.length} vs ${b.length} bytes`);

const n = (Math.min(a.length, b.length) - HEADER) >> 1;
let maxDiff = 0, sumSq = 0, peak = 0, at = -1;
for (let i = 0; i < n; i++) {
    const x = a.readInt16LE(HEADER + i * 2);
    const y = b.readInt16LE(HEADER + i * 2);
    const d = Math.abs(x - y);
    if (d > maxDiff) { maxDiff = d; at = i; }
    sumSq += d * d;
    peak = Math.max(peak, Math.abs(x), Math.abs(y));
}
const rms = Math.sqrt(sumSq / n);

console.log(`${n} samples, peak ${peak}`);
console.log(`max difference ${maxDiff} (sample ${at}) = ${(100 * maxDiff / peak).toFixed(3)}% of peak`);
console.log(`rms difference ${rms.toFixed(3)} = ${(100 * rms / peak).toFixed(4)}% of peak`);
