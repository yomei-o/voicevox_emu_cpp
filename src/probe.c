// probe.c - does voicevox_onnxruntime load and initialise inside the emulator?
//
// No model, no audio.  This only asks whether the dynamic loader, the C++
// runtime's static initialisers, and ORT's own global setup survive.  Each
// step prints before and after, so a fault names the step it died in.
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include "onnxruntime_c_api.h"

static const OrtApi* g_api;

static void check(OrtStatus* st, const char* what) {
    if (!st) {
        printf("ok    %s\n", what);
        return;
    }
    printf("FAIL  %s: %s\n", what, g_api->GetErrorMessage(st));
    g_api->ReleaseStatus(st);
    exit(1);
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "libvoicevox_onnxruntime.so.1.17.3";

    printf("step  dlopen %s\n", path);
    fflush(stdout);
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        printf("FAIL  dlopen: %s\n", dlerror());
        return 1;
    }
    printf("ok    dlopen\n");
    fflush(stdout);

    const OrtApiBase* (*get_base)(void) = (const OrtApiBase* (*)(void))dlsym(h, "OrtGetApiBase");
    if (!get_base) {
        printf("FAIL  dlsym OrtGetApiBase: %s\n", dlerror());
        return 1;
    }
    printf("ok    dlsym OrtGetApiBase\n");
    fflush(stdout);

    const OrtApiBase* base = get_base();
    printf("ok    OrtGetApiBase -> version %s\n", base->GetVersionString());
    fflush(stdout);

    g_api = base->GetApi(ORT_API_VERSION);
    if (!g_api) {
        printf("FAIL  GetApi(%d) returned NULL\n", ORT_API_VERSION);
        return 1;
    }
    printf("ok    GetApi(%d)\n", ORT_API_VERSION);
    fflush(stdout);

    OrtEnv* env = NULL;
    check(g_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "probe", &env), "CreateEnv");
    fflush(stdout);

    OrtSessionOptions* so = NULL;
    check(g_api->CreateSessionOptions(&so), "CreateSessionOptions");

    // Single threaded and no spinning: the emulator schedules guest threads
    // cooperatively, so a spinning thread pool would never yield.
    check(g_api->SetIntraOpNumThreads(so, 1), "SetIntraOpNumThreads(1)");
    check(g_api->SetInterOpNumThreads(so, 1), "SetInterOpNumThreads(1)");
    check(g_api->SetSessionExecutionMode(so, ORT_SEQUENTIAL), "SetSessionExecutionMode(SEQUENTIAL)");
    check(g_api->AddSessionConfigEntry(so, "session.intra_op.allow_spinning", "0"),
          "allow_spinning=0");
    check(g_api->AddSessionConfigEntry(so, "session.inter_op.allow_spinning", "0"),
          "inter_op allow_spinning=0");

    // Does this runtime accept the encrypted model format at all?  Asking for
    // the config entry is not the same as loading one, but a runtime that
    // rejects the key outright is not the patched build.
    check(g_api->AddSessionConfigEntry(so, "session.use_vv_bin", "1"), "session.use_vv_bin=1");

    char** providers = NULL;
    int nproviders = 0;
    check(g_api->GetAvailableProviders(&providers, &nproviders), "GetAvailableProviders");
    for (int i = 0; i < nproviders; i++) printf("      provider[%d] = %s\n", i, providers[i]);
    g_api->ReleaseAvailableProviders(providers, nproviders);

    OrtAllocator* alloc = NULL;
    check(g_api->GetAllocatorWithDefaultOptions(&alloc), "GetAllocatorWithDefaultOptions");

    // A real allocation and a real tensor, so the memory path is exercised.
    int64_t shape[2] = {2, 3};
    float data[6] = {1, 2, 3, 4, 5, 6};
    OrtMemoryInfo* mi = NULL;
    check(g_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi),
          "CreateCpuMemoryInfo");
    OrtValue* t = NULL;
    check(g_api->CreateTensorWithDataAsOrtValue(mi, data, sizeof data, shape, 2,
                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t),
          "CreateTensorWithDataAsOrtValue");
    float* back = NULL;
    check(g_api->GetTensorMutableData(t, (void**)&back), "GetTensorMutableData");
    printf("      tensor[0..5] = %g %g %g %g %g %g\n", back[0], back[1], back[2], back[3], back[4],
           back[5]);

    // Optional second argument: a plain, unencrypted .onnx.  This is the
    // control for the vv_bin failure - if an ordinary model parses and builds
    // a session here, ORT's own machinery is sound under emulation and only
    // the encrypted path is in question.
    if (argc > 2) {
        printf("step  CreateSession %s\n", argv[2]);
        fflush(stdout);
        OrtSession* sess = NULL;
        check(g_api->CreateSession(env, argv[2], so, &sess), "CreateSession");
        size_t n_in = 0, n_out = 0;
        check(g_api->SessionGetInputCount(sess, &n_in), "SessionGetInputCount");
        check(g_api->SessionGetOutputCount(sess, &n_out), "SessionGetOutputCount");
        printf("      %zu inputs, %zu outputs\n", n_in, n_out);
        for (size_t i = 0; i < n_in; i++) {
            char* name = NULL;
            check(g_api->SessionGetInputName(sess, i, alloc, &name), "SessionGetInputName");
            printf("      in[%zu]  %s\n", i, name);
            alloc->Free(alloc, name);
        }
        for (size_t i = 0; i < n_out; i++) {
            char* name = NULL;
            check(g_api->SessionGetOutputName(sess, i, alloc, &name), "SessionGetOutputName");
            printf("      out[%zu] %s\n", i, name);
            alloc->Free(alloc, name);
        }
        g_api->ReleaseSession(sess);
    }

    g_api->ReleaseValue(t);
    g_api->ReleaseMemoryInfo(mi);
    g_api->ReleaseSessionOptions(so);
    g_api->ReleaseEnv(env);
    printf("ok    teardown\n");
    printf("PROBE OK\n");
    return 0;
}
