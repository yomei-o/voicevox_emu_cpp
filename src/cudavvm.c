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

#include "voicevox_core.h"

static void step(const char* what) {
    printf("step  %s\n", what);
    fflush(stdout);
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

    step("voicevox_synthesizer_new  <- acceleration = GPU");
    VoicevoxInitializeOptions iopts = voicevox_make_default_initialize_options();
    iopts.acceleration_mode = VOICEVOX_ACCELERATION_MODE_GPU;
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
    step("voicevox_synthesizer_tts  <- every kernel the pipeline uses");
    const char* text = argc > 5 ? argv[5] : "あ";
    const char* out_path = argc > 6 ? argv[6] : "shim.wav";
    VoicevoxTtsOptions topts = voicevox_make_default_tts_options();
    uintptr_t len = 0;
    uint8_t* wav = NULL;
    VoicevoxResultCode r = voicevox_synthesizer_tts(syn, text, style, topts, &len, &wav);
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
