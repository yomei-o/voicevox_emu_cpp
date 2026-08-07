// cudavvm.c - the whole VOICEVOX pipeline through the CUDA provider, with no GPU.
//
// cudaprobe answers the question for a plain .onnx: four distinct kernels.  The
// number that decides the CUDA-shim idea is the one for the *vocoder*, which is
// the expensive part and which only exists inside the encrypted model.  So this
// runs the real pipeline - the same published API, the same CORE - with
// acceleration set to GPU and the CUDA build of the runtime underneath, against
// the stand-in libraries from tools/make_cuda_stubs.sh.
//
// Nothing computes: cuBLAS and cuDNN return success without touching the data,
// so whatever comes out the far end is noise.  The output is the list of kernel
// names `cudaLaunchKernel` printed on the way, which is what a shim would have
// to implement.
//
//     cudavvm <ort.so> <dict-dir> <model.vvm> [style]
//
// The decryption still happens inside the runtime, exactly as it does when the
// thing is used for real; nothing here reads a model back out.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "vvhostcall.h"

#include "voicevox_core.h"

// Each step says how long the one before it took.  Module-level profiling puts
// Open JTalk inside libvoicevox_core, where it is indistinguishable from the
// decryption and the rest of CORE - and "which phase" separates them without
// needing to.
static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static double step_started;

static void step(const char* what) {
    if (step_started > 0) printf("      ... %.1f s\n", now_seconds() - step_started);
    printf("step  %s\n", what);
    fflush(stdout);
    step_started = now_seconds();
}

static int check(VoicevoxResultCode r, const char* what) {
    if (r == VOICEVOX_RESULT_OK) {
        printf("ok    %s\n", what);
        fflush(stdout);
        return 0;
    }
    printf("FAIL  %s: %d %s\n", what, (int)r, voicevox_error_result_to_message(r));
    fflush(stdout);
    return 1;
}

int main(int argc, char** argv) {
    const char* ort = argc > 1 ? argv[1] : "./libvoicevox_onnxruntime.so.1.17.3";
    const char* dict = argc > 2 ? argv[2] : "./open_jtalk_dic_utf_8-1.11";
    const char* vvm = argc > 3 ? argv[3] : "./0.vvm";
    uint32_t style = argc > 4 ? (uint32_t)atoi(argv[4]) : 3;

    step("voicevox_onnxruntime_load_once  <- the CUDA build");
    VoicevoxLoadOnnxruntimeOptions oopts = voicevox_make_default_load_onnxruntime_options();
    oopts.filename = ort;
    const VoicevoxOnnxruntime* rt = NULL;
    if (check(voicevox_onnxruntime_load_once(oopts, &rt), "load_once")) return 1;

    {
        char* json = NULL;
        if (voicevox_onnxruntime_create_supported_devices_json(rt, &json) ==
            VOICEVOX_RESULT_OK) {
            printf("      devices %s\n", json);
            voicevox_json_free(json);
        }
    }

    step("voicevox_open_jtalk_rc_new");
    OpenJtalkRc* ojt = NULL;
    if (check(voicevox_open_jtalk_rc_new(dict, &ojt), "open_jtalk_rc_new")) return 1;

    // VVSTUB_CPU=1 asks for the CPU provider instead, which is how the shim
    // gets a like-for-like baseline: the same binary, the same model, the same
    // utterance, with the arithmetic going somewhere else.
    int cpu_mode = getenv("VVSTUB_CPU") && *getenv("VVSTUB_CPU") != '0';
    step(cpu_mode ? "voicevox_synthesizer_new  <- acceleration = CPU"
                  : "voicevox_synthesizer_new  <- acceleration = GPU");
    VoicevoxInitializeOptions iopts = voicevox_make_default_initialize_options();
    iopts.acceleration_mode = cpu_mode ? VOICEVOX_ACCELERATION_MODE_CPU
                                       : VOICEVOX_ACCELERATION_MODE_GPU;
    VoicevoxSynthesizer* syn = NULL;
    if (check(voicevox_synthesizer_new(rt, ojt, iopts, &syn), "synthesizer_new")) return 1;
    printf("      is_gpu_mode = %s\n",
           voicevox_synthesizer_is_gpu_mode(syn) ? "true" : "false");

    step("voicevox_voice_model_file_open");
    VoicevoxVoiceModelFile* model = NULL;
    if (check(voicevox_voice_model_file_open(vvm, &model), "voice_model_file_open")) return 1;

    step("voicevox_synthesizer_load_voice_model  <- decrypt and build every session");
    if (check(voicevox_synthesizer_load_voice_model(syn, model), "load_voice_model")) return 1;
    voicevox_voice_model_file_delete(model);

    // Everything above only builds sessions.  The kernels of interest launch
    // here, and the ones from the vocoder are the ones that matter.
    // VVSNAPSHOT=<path>: dump the emulator's state here, now, with the sessions
    // built and before a single kernel has launched.  That is the point a
    // resume would start from, so it is the point worth measuring.  Outside the
    // emulator the syscall answers ENOSYS and nothing happens.
    {
        const char* snap = getenv("VVSNAPSHOT");
        if (snap) {
            long a[VVHOST_SLOTS] = {(long)snap};
            long n = vvhost(VVH_SNAPSHOT, a);
            printf("      snapshot %ld bytes -> %s\n", n, snap);
        }
    }

    step("voicevox_synthesizer_tts  <- every kernel the pipeline uses");
    // `@path` reads the text from a file.  Passing it as an argument works on
    // Linux and does not survive the trip through a Windows command line: the
    // emulator's guest got "あ" as mojibake and the CORE rejected it as invalid
    // UTF-8, which is the right answer to the wrong bytes.
    const char* text = argc > 5 ? argv[5] : "あ";
    char text_buf[4096];
    if (text[0] == '@') {
        FILE* tf = fopen(text + 1, "rb");
        if (!tf) {
            printf("      cannot read text file %s\n", text + 1);
            return 1;
        }
        size_t n = fread(text_buf, 1, sizeof text_buf - 1, tf);
        fclose(tf);
        while (n && (text_buf[n - 1] == '\n' || text_buf[n - 1] == '\r')) n--;
        text_buf[n] = '\0';
        text = text_buf;
    }
    const char* out_path = argc > 6 ? argv[6] : "shim.wav";
    VoicevoxTtsOptions topts = voicevox_make_default_tts_options();
    uintptr_t len = 0;
    uint8_t* wav = NULL;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    VoicevoxResultCode r = voicevox_synthesizer_tts(syn, text, style, topts, &len, &wav);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("      tts took %.3f s\n",
           (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);
    if (r == VOICEVOX_RESULT_OK) {
        printf("      produced %zu bytes\n", (size_t)len);
        FILE* f = fopen(out_path, "wb");
        if (f) {
            fwrite(wav, 1, (size_t)len, f);
            fclose(f);
            printf("      wrote %s\n", out_path);
        }
        voicevox_wav_free(wav);
    } else {
        printf("      tts returned %d %s\n", (int)r, voicevox_error_result_to_message(r));
    }

    printf("CUDAVVM DONE\n");
    return 0;
}
