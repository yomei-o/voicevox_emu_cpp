# Running the CUDA build with no GPU, and what it would take to make it compute

An idea, and the measurements that decide it.

**The idea.** VOICEVOX's CUDA runtime offloads its arithmetic through a
documented C boundary — libcudart, cuBLAS, cuDNN. Run *that* build inside the
emulator, intercept those calls, and do the maths natively. The decryption
stays where it belongs, inside the runtime; only the arithmetic moves.

**Why this boundary and no other.** The CPU build of
`libvoicevox_onnxruntime.so.1.17.3` exports exactly **three** symbols:

    OrtGetApiBase
    OrtSessionOptionsAppendExecutionProvider_CPU
    VERS_1.17.3

MLAS, the kernels, all of it is hidden. There is nothing inside the CPU build to
hook. The CUDA build's calls into libcudart and friends are *imports*, so they
go through the PLT and are visible. It is the only seam there is.

**And nothing here needs reverse engineering.** ONNX Runtime's CUDA kernels are
upstream open source, and the runtime is MIT. What VOICEVOX patched is the
`vv_bin` path, and that stays inside the emulated runtime where it belongs.

## What it takes: measured, not guessed

`tools/make_cuda_stubs.sh` builds stand-in libcudart / libcublas / libcublasLt /
libcudnn / libcufft — every entry point the provider imports, returning success
and computing nothing. `src/cudastub.c` makes a handful real: the allocator
hands out host memory, the copies are `memcpy`, and `__cudaRegisterFunction`
records each kernel's name so `cudaLaunchKernel` can print which one ran.

Then `src/cudavvm.c` runs the whole published pipeline against them — the same
CORE, the same API, `acceleration_mode = GPU`.

**It works.** On a machine with no GPU and no CUDA installed:

```
      devices {"cpu":true,"cuda":true,"dml":false}
  INFO voicevox_core::synthesizer: GPUをテストします:
  INFO voicevox_core::synthesizer:   * CUDA (device_id=0): OK
  INFO voicevox_core::synthesizer: CUDA (device_id=0)を利用します
ok    synthesizer_new          is_gpu_mode = true
ok    load_voice_model         <- decrypted, every session built
ok    tts                      25132 bytes (the right size; the values are noise)
[cuda] 4757 kernels registered, 377 launches
```

### The numbers

| | |
| --- | --- |
| CUDA entry points the provider *imports* | 149 |
| kernels it *registers* at load | 4757 |
| — of those, **launched by a real utterance** | **43** specialised, **20 families** |
| cuDNN functions actually called | 15 |
| cuBLAS functions actually called | 5 |

The gap between 4757 and 43 is the whole point: registering is not running.

### The twenty kernel families

    _UnaryElementWise            _BinaryElementWise         _TenaryElementWise
    _BinaryElementWiseSimple     _BinaryElementWiseRhsPerChannelBatch1
    _BinaryElementWiseRhsPerChannelBatchN
    _GatherKernel                _ConcatKernel              _ConcatKernelSameConcatDim
    _SplitKernelSameSplitDim     _SliceKernel               _ScatterNDKernel
    TransposeKernel              ExpandKernel               ExpandKernel2D
    RangeKernel                  _Fill                      _FillFromDataPtrKernel
    reduce_matrix_columns_kernel softmax_warp_forward

The element-wise ones are one shape each, parameterised by an operator: Add,
Sub, Mul, Div, Pow, Sqrt, Tanh, Sigmoid, Relu, LeakyRelu, QuickGelu. So the work
is roughly twenty loop shapes and a dozen scalar functions.

### The eight that actually compute

Everything else is bookkeeping — creating and filling descriptors, which is
numerous (291 tensor descriptors for one utterance) and trivial.

    cudnnConvolutionForward           the convolutions
    cudnnConvolutionBackwardData      the transposed convolutions (the vocoder)
    cudnnFindConvolutionBackwardDataAlgorithmEx
    cudnnAddTensor                    bias
    cublasSgemm_v2                    the matrix multiplies
    cublasSgemmStridedBatched
    cublasSgeam                       transpose/scale
    cublasLtCreate                    (created, never used to compute)

Eigen does all of these.

## The reference, on real hardware

`colab/voicevox_cuda_reference.ipynb` runs the same two programs on a Colab T4
against the real libraries, and it works: the CUDA provider comes up, the model
decrypts, and the audio is audible. So the shim has an unambiguous target rather
than an argument.

**The CPU and the GPU agree exactly.** `predict_duration.onnx` answers with the
same sixteen values to every digit printed, from the CPU provider on a desktop
and the CUDA provider on a T4 - two paths that share no arithmetic:

    1.129583  0.202718  0.191819  0.100126  0.061151  0.043386  0.070718 ...

They are in `colab/predict_duration_reference.txt`. That is the shim's target,
and it is not a matter of opinion: a shim that gets these is right, one that does
not is wrong, and the index of the first disagreement says where to look.

**The shim reproduces all sixteen**, to every digit. `tools/check_shim.sh` runs
it. And having done that it goes on to make the audio:

| "ずんだもんなのだ", 1.45 s | largest sample difference | against a peak of 12988 |
| --- | --- | --- |
| the shim, against the T4 | 2 | 0.015 % |
| the shim, against `x86emu` | 7 | 0.054 % |

and for "あ": 3 of 3912 against the T4. `web/sample/shim_*.wav` are its output,
kept beside the T4's and the emulator's. All 377 launches an utterance makes are
handled natively; nothing falls through to a stub.

So the answer to "eight functions and twenty loop shapes" is: yes, that was the
whole of it. The compute surface is `src/cudnn_real.cpp` (convolution, its
transpose, bias), `src/cublas_real.cpp` (Sgemm, its batched form, Sgeam) and
`src/cudakernels.c` (twenty kernel families over five element types) - about
1400 lines, all of it plain loops and Eigen.

### The two bugs worth writing down

Both were found the same way, and neither would have been findable without a
reference that is exact.

**A shift of six.** With the convolutions running but Gather and Sgeam still
empty, `predict_duration` answered structured nonsense. Once those were filled
in, the output became the reference *shifted by six positions* - our `[0..5]`
were its `[6..11]`, to every digit. That is a legible failure: an exact shift
means the arithmetic is right and the indexing is not. ONNX Runtime lays a 1-D
convolution out as `[N, C, W, 1]` - the extent is in the *height* slot, and the
width is the degenerate one. Reading pad/stride/dilation at the last slot got
`pad = 0` where the model asked for `2`, and the three convolutions shifted by
two apiece. Noise would have said nothing; an exact shift said where to look.

**A unary division.** The vocoder then produced *silence* - and the sample count
was exactly right, which said the duration model was correct and the vocoder was
not. `VVSTUB_STATS=1` prints each kernel's output range, and the trace showed the
values multiplying by about a hundred per residual block until they reached
infinity. The cause: `OP_Div` appears as a **unary** operator, 22 times, once per
reduction. It is not `1/x`. It is how ONNX Runtime finishes a `ReduceMean` - the
reduction sums and this divides by the count, which travels inside the functor
the way LeakyRelu's slope does. Reading it as a reciprocal turned every mean into
its inverse, so every LayerNorm scaled up instead of down. The ranges said so
plainly: the reduction's mean was exactly 64 times its input's, which is a sum of
64 things, not a reciprocal of anything.

And the audio it makes is the audio everything else here makes. The same
utterance, through four very different machines:

| "あ", 0.52 s of audio | largest sample difference | against a peak of 3912 |
| --- | --- | --- |
| a Tesla T4, against native CPU | 2 | 0.051 % |
| a Tesla T4, against `x86emu` | 2 | 0.051 % |
| a Tesla T4, against the browser | 2 | 0.051 % |

and for "ずんだもんなのだ", 1.45 s: 1 of 12988 against native, 8 against the
emulator. Under a tenth of a percent every way you cut it - which is the last
thing this project needed to say. Not "it ran under an emulator", but **the
same answer as the dedicated hardware**, from a CPU that is software, in a
browser tab.

`web/sample/gpu_*.wav` are the T4's own output, kept beside the emulator's — and
`web/sample/shim_*.wav` are what this document is about: the same audio, from the
CUDA build, on a machine with no GPU and no CUDA installed.

Four things went wrong getting there, and all four failed in a way that pointed
somewhere else. They are written down because each would cost an hour again:

| what happened | what it looked like |
| --- | --- |
| a `!` line's `{name}` substitution did not happen, so `cp` copied nothing | `cannot open shared object file`, several cells later |
| the provider is linked against cuDNN **8**, a current Colab ships 9 | `libcudnn.so.8: cannot open shared object file` |
| Colab's toolkit can be newer than its driver | `CUDA failure 35: driver version is insufficient` |
| **setting `LD_LIBRARY_PATH` instead of adding to it hid `/usr/lib64-nvidia`** | the same error 35 - with no driver at all, `cudaDriverGetVersion` answers 0 |

The last one is the one to remember: error 35 reads as a version mismatch and is
also what "there is no driver on the path" looks like. The cell that asks
`cudaRuntimeGetVersion` and `cudaDriverGetVersion` directly settled it in one
line, after three wrong guesses.

## So: is it a weekend or a year?

A weekend. The compute surface was **eight functions and twenty loop shapes**,
exactly as the measurement said, and filling them in produces the T4's audio to
within 2 samples of 12988. The rest is descriptor plumbing and stubs that return
success.

Two of the three open questions are now answered.

1. **Are the shapes right?** *Yes.* This was a real worry: with cuBLAS and cuDNN
   returning success without computing, the values were noise, and noise can take
   a different branch than real data would. It did not. The kernel list measured
   against nonsense is the same list the working shim uses — 43 specialisations,
   20 families, 377 launches for an utterance.

2. **Is it faster?** *Not on its own.* Same machine, same library, same
   utterance, "ずんだもんなのだ":

   | | tts |
   | --- | --- |
   | the CPU provider (MLAS) | 1.1 s |
   | the shim | ~10 s |

   Which is the expected answer and worth stating plainly: MLAS is hand-written
   AVX across every core, and this is im2col plus scalar C loops. A tenth of the
   speed for a thousandth of the code is the trade that was made on purpose.

   The comparison that matters is the other one. Synthesis *inside the emulator*
   takes hours; this moves all of the arithmetic out of it for ten seconds of
   native work. That is the win the whole design was for — but see the next
   point, because it has not been measured yet.

3. **Does it run under the emulator?** *Still unknown, and this is now the only
   thing left.* Everything above was measured natively on Linux, which was the
   right way round: the runtime behaves the same either way and iteration is
   seconds instead of minutes. The emulator needs one new thing — hooking an ELF
   guest's imports. `Emulator::dispatch_hook` is address-based and already
   exists; wiring it to a PLT is new work. And 377 launches is 377 round trips
   across the emulator's memory, which is the part that could still spoil it.

## Could this make the browser demo bearable?

That is the question the whole exercise is really for, and three of the four
things it turns on are now measured.

**How much of the work is arithmetic? Nearly all of it.** The CUDA provider
hands every bit of its arithmetic across this boundary, so timing the shim
answers it exactly (`VVSTUB_TIME=1`):

| "ずんだもんなのだ" | |
| --- | --- |
| whole run | 4.97 s |
| — synthesis | 3.71 s |
| — — inside the shim: **the arithmetic** | **3.70 s** |
| — — ONNX Runtime's own plumbing | 0.01 s |
| everything else: model load, decrypt, session build, OpenJTalk | 1.27 s |

So of a run, **99.7 % of the synthesis is arithmetic** that this design moves
out of the interpreter, and what stays behind is the 1.27 s of setup. That is
the ceiling, and it is a high one.

**Is the download impossible? No, after `tools/slim_provider.sh`.** The provider
is 440 MB and 419 MB of it is `.nv_fatbin` — compiled device code for all 4757
kernels, of which the shim executes none. Zeroing those bytes in place keeps
every offset in the file correct, and the provider still registers 4757 kernels,
still launches 377, and still makes audio within 2 of 12988 of the T4:

    440 MB  ->  9.3 MB gzipped

**Does it fit in a tab? Not yet.** `Memory::map` allocates every page of a
segment up front and the loader then writes the whole segment, so those 419 MB
of zeros still cost 419 MB of guest pages. Lazy pages — record the range,
allocate on first touch — would fix it, and would help every other large
mapping too. That is a bounded change to the emulator core.

**Would it be fast? Unknown, and this is the honest part.** The arithmetic would
run as compiled WebAssembly rather than interpreted x86, which is the whole
point. But the 1.27 s of setup stays emulated, and at the browser's ~6 M
instructions a second that is minutes, not seconds. So the shape of the answer
is: **from about 100 minutes to something on the order of a few** — an enormous
improvement and still not instant. The error bar on that stays wide until the
PLT hooking exists and it can be *measured* instead of estimated.

And one more thing to weigh against it: the shim is competitive per core but it
is single-threaded, where MLAS is not. Same utterance, this machine, eight
cores: MLAS 1.1 s against the shim's 3.7 s. Per core the shim is ahead; on wall
clock it is not, and a browser tab is where that gap is hardest to close.

## Where the line is

Unlike a JIT — which speeds the emulator up without the weights ever being used
outside it — this design has native code operating on the decrypted weights, in
host memory, on purpose. Whether that reads as "the emulator provides a device"
or as something the voice-model terms forbid is a judgement call, and it is the
owner's to make. It is not a technical question and this document does not
answer it.

Shipping it in the **public demo page** is a further step again, and a larger
one: there the decrypted weights would sit in a browser tab's linear memory,
reachable by anything on the page, with a documented layout. That deserves its
own decision rather than following from this one.

Nothing here has been used to extract anything: the measurements above run a
plain `.onnx` and a real utterance whose output is deliberately meaningless.
