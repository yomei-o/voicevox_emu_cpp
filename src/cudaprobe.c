// cudaprobe.c - which of ONNX Runtime's CUDA kernels does a model actually use?
//
// The CUDA build of voicevox_onnxruntime registers 2793 kernels and imports 149
// entry points across libcudart, cuBLAS, cuDNN and cuFFT.  Registering is not
// running, though, and importing is not calling: a convolution-shaped model may
// spend all its time inside cuDNN and never launch a kernel of ORT's own.  The
// difference decides whether "hook the CUDA calls and do the maths natively" is
// a weekend or a year, so it is worth measuring rather than arguing about.
//
// This loads the CUDA provider against the stand-in libraries from
// tools/make_cuda_stubs.sh - no GPU, no CUDA installed, nothing computed - and
// lets `cudaLaunchKernel` print the name of every kernel that gets launched.
//
//     cudaprobe <ort.so> <model.onnx>
//
// Nothing here touches an encrypted model: it runs a plain .onnx, which is the
// point - the question is about the runtime, not about the voice.
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnxruntime_c_api.h"

static const OrtApi* g_api;

static void step(const char* what) {
    printf("step  %s\n", what);
    fflush(stdout);
}

static int check(OrtStatus* st, const char* what) {
    if (!st) {
        printf("ok    %s\n", what);
        fflush(stdout);
        return 0;
    }
    printf("FAIL  %s: %s\n", what, g_api->GetErrorMessage(st));
    fflush(stdout);
    g_api->ReleaseStatus(st);
    return 1;
}

int main(int argc, char** argv) {
    const char* ort_path = argc > 1 ? argv[1] : "/opt/vv/libvoicevox_onnxruntime.so.1.17.3";
    const char* model = argc > 2 ? argv[2] : "/opt/vv/predict_duration.onnx";

    step("dlopen the CUDA build");
    void* h = dlopen(ort_path, RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        printf("FAIL  dlopen: %s\n", dlerror());
        return 1;
    }
    printf("ok    dlopen\n");

    const OrtApiBase* (*get_base)(void) = dlsym(h, "OrtGetApiBase");
    if (!get_base) {
        printf("FAIL  dlsym OrtGetApiBase\n");
        return 1;
    }
    g_api = get_base()->GetApi(ORT_API_VERSION);
    printf("ok    OrtGetApiBase -> %s\n", get_base()->GetVersionString());

    OrtEnv* env = NULL;
    step("CreateEnv");
    if (check(g_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "cudaprobe", &env), "CreateEnv"))
        return 1;

    OrtSessionOptions* opts = NULL;
    step("CreateSessionOptions");
    if (check(g_api->CreateSessionOptions(&opts), "CreateSessionOptions")) return 1;

    // The CUDA provider, through the plain C entry point the build exports.
    step("append the CUDA execution provider");
    OrtStatus* (*append_cuda)(OrtSessionOptions*, int) =
        dlsym(h, "OrtSessionOptionsAppendExecutionProvider_CUDA");
    if (!append_cuda) {
        printf("FAIL  this build has no CUDA provider\n");
        return 1;
    }
    if (check(append_cuda(opts, 0), "AppendExecutionProvider_CUDA")) return 1;

    step("CreateSession  <- the provider initialises here");
    OrtSession* session = NULL;
    if (check(g_api->CreateSession(env, model, opts, &session), "CreateSession")) return 1;

    // What the session decided about the graph, before running anything.
    size_t n_in = 0, n_out = 0;
    g_api->SessionGetInputCount(session, &n_in);
    g_api->SessionGetOutputCount(session, &n_out);
    printf("      %zu inputs, %zu outputs\n", n_in, n_out);

    OrtAllocator* alloc = NULL;
    g_api->GetAllocatorWithDefaultOptions(&alloc);
    char* in_names[8] = {0};
    char* out_names[8] = {0};
    for (size_t i = 0; i < n_in && i < 8; i++)
        g_api->SessionGetInputName(session, i, alloc, &in_names[i]);
    for (size_t i = 0; i < n_out && i < 8; i++)
        g_api->SessionGetOutputName(session, i, alloc, &out_names[i]);
    for (size_t i = 0; i < n_in && i < 8; i++) printf("      input[%zu]  %s\n", i, in_names[i]);
    for (size_t i = 0; i < n_out && i < 8; i++) printf("      output[%zu] %s\n", i, out_names[i]);

    // predict_duration takes a phoneme sequence and a speaker id.  The values do
    // not matter - nothing here computes an answer - but the shapes do, because
    // that is what decides which kernels get chosen.
    step("Run  <- the kernels launch here");
    OrtMemoryInfo* mem = NULL;
    g_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem);

    static int64_t phonemes[16];
    for (int i = 0; i < 16; i++) phonemes[i] = i % 40;
    static int64_t speaker[1] = {0};
    int64_t shape_p[1] = {16}, shape_s[1] = {1};

    OrtValue* v_in[2] = {0};
    g_api->CreateTensorWithDataAsOrtValue(mem, phonemes, sizeof phonemes, shape_p, 1,
                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &v_in[0]);
    g_api->CreateTensorWithDataAsOrtValue(mem, speaker, sizeof speaker, shape_s, 1,
                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &v_in[1]);

    OrtValue* v_out[1] = {0};
    const char* ins[2] = {in_names[0], in_names[1]};
    const char* outs[1] = {out_names[0]};
    OrtStatus* st = g_api->Run(session, NULL, ins, (const OrtValue* const*)v_in,
                               n_in < 2 ? n_in : 2, outs, 1, v_out);
    if (st) {
        printf("      Run: %s\n", g_api->GetErrorMessage(st));
        g_api->ReleaseStatus(st);
    } else {
        printf("ok    Run\n");
    }

    printf("CUDAPROBE DONE\n");
    return 0;
}
