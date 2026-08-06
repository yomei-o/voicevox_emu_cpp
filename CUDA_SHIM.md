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

`web/sample/gpu_*.wav` are the T4's own output, kept beside the emulator's.

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

Neither, but far closer to the first than the 4757 kernels suggested. The
compute surface is **eight functions and twenty loop shapes**. The rest is
descriptor plumbing and stubs that return success.

What is left to find out, in order:

1. **Does it run under the emulator?** Everything above was measured natively on
   Linux, which is the right way round — the runtime's behaviour is the same
   either way and iteration is seconds instead of minutes. The emulator needs
   one new thing: hooking an ELF guest's imports. `Emulator::dispatch_hook` is
   address-based and already exists; wiring it to a PLT is new work.
2. **Are the shapes right?** With cuBLAS and cuDNN returning success without
   computing, the values are noise, and noise can take a different branch than
   real data would. The kernel list may be incomplete until the maths is real.
3. **Is it actually faster?** The point of the exercise. Native Eigen
   convolutions against an emulated CPU should be enormous, but the tensors
   still cross the emulator's memory on every call, and 377 launches is 377
   round trips.

## Where the line is

Unlike a JIT — which speeds the emulator up without the weights ever being used
outside it — this design has native code operating on the decrypted weights, in
host memory, on purpose. Whether that reads as "the emulator provides a device"
or as something the voice-model terms forbid is a judgement call, and it is the
owner's to make. It is not a technical question and this document does not
answer it.

Nothing here has been used to extract anything: the measurements above run a
plain `.onnx` and a real utterance whose output is deliberately meaningless.
