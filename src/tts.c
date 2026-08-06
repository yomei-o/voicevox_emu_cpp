// tts.c - the official VOICEVOX CORE, inside the emulator, writing a WAV.
//
// Every step prints and flushes before it runs, and the elapsed seconds after,
// so a run that takes an hour still says where it is and a fault names the step.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "voicevox_core.h"

// Also built natively for Windows, as the control: the same model through the
// same official binaries, with no emulator in the way.  If that fails too,
// the problem is not the emulator.
static double now(void) {
#if defined(_WIN32)
    return (double)clock() / CLOCKS_PER_SEC;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

static double t_step;

static void step(const char* what) {
    printf("step  %s\n", what);
    fflush(stdout);
    t_step = now();
}

static void ok(void) {
    printf("      ... %.1f s\n", now() - t_step);
    fflush(stdout);
}

static void check(VoicevoxResultCode r, const char* what) {
    if (r == VOICEVOX_RESULT_OK) {
        ok();
        return;
    }
    printf("FAIL  %s: %d %s\n", what, (int)r, voicevox_error_result_to_message(r));
    fflush(stdout);
    exit(1);
}

int main(int argc, char** argv) {
    const char* ort_path = argc > 1 ? argv[1] : "/opt/vv/libvoicevox_onnxruntime.so.1.17.3";
    const char* dict = argc > 2 ? argv[2] : "/opt/vv/open_jtalk_dic_utf_8-1.11";
    const char* vvm = argc > 3 ? argv[3] : "/opt/vv/0.vvm";
    const char* text = argc > 4 ? argv[4] : "ずんだもんなのだ";
    uint32_t style = argc > 5 ? (uint32_t)atoi(argv[5]) : 3;  // ずんだもん ノーマル
    const char* out = argc > 6 ? argv[6] : "/opt/vv/out.wav";

    double t0 = now();
    printf("core  %s\n", voicevox_get_version());
    fflush(stdout);

    step("voicevox_onnxruntime_load_once");
    VoicevoxLoadOnnxruntimeOptions oopts = voicevox_make_default_load_onnxruntime_options();
    oopts.filename = ort_path;
    const VoicevoxOnnxruntime* ort = NULL;
    check(voicevox_onnxruntime_load_once(oopts, &ort), "load_once");

    step("voicevox_open_jtalk_rc_new");
    OpenJtalkRc* ojt = NULL;
    check(voicevox_open_jtalk_rc_new(dict, &ojt), "open_jtalk");

    step("voicevox_synthesizer_new");
    VoicevoxInitializeOptions iopts = voicevox_make_default_initialize_options();
    iopts.acceleration_mode = VOICEVOX_ACCELERATION_MODE_CPU;
    iopts.cpu_num_threads = 1;  // the emulator schedules guest threads cooperatively
    VoicevoxSynthesizer* syn = NULL;
    check(voicevox_synthesizer_new(ort, ojt, iopts, &syn), "synthesizer_new");

    // Control: the guest's own view of the model bytes.  A decrypt that comes
    // out as garbage looks identical to a file the emulated read path handed
    // back wrong, and this tells the two apart before anything harder is tried.
    step("checksum the model file as the guest sees it");
    {
        FILE* mf = fopen(vvm, "rb");
        if (!mf) {
            printf("FAIL  fopen %s\n", vvm);
            return 1;
        }
        uint64_t h = 1469598103934665603ull, total = 0;  // FNV-1a
        static unsigned char buf[65536];
        size_t got;
        while ((got = fread(buf, 1, sizeof buf, mf)) > 0) {
            for (size_t i = 0; i < got; i++) h = (h ^ buf[i]) * 1099511628211ull;
            total += got;
        }
        fclose(mf);
        printf("      %llu bytes, fnv1a %016llx\n", (unsigned long long)total,
               (unsigned long long)h);
    }
    ok();

    step("voicevox_voice_model_file_open");
    VoicevoxVoiceModelFile* model = NULL;
    check(voicevox_voice_model_file_open(vvm, &model), "model_open");

    // This is the one that matters: the encrypted vv_bin payload goes into the
    // patched ONNX Runtime and comes back as a live session, or it does not.
    step("voicevox_synthesizer_load_voice_model  <- vv_bin decrypt + session init");
    check(voicevox_synthesizer_load_voice_model(syn, model), "load_voice_model");
    voicevox_voice_model_file_delete(model);

    printf("text  %s  (style %u)\n", text, style);
    step("voicevox_synthesizer_tts  <- inference");
    VoicevoxTtsOptions topts = voicevox_make_default_tts_options();
    uintptr_t wav_len = 0;
    uint8_t* wav = NULL;
    check(voicevox_synthesizer_tts(syn, text, style, topts, &wav_len, &wav), "tts");

    FILE* f = fopen(out, "wb");
    if (!f) {
        printf("FAIL  fopen %s\n", out);
        return 1;
    }
    fwrite(wav, 1, wav_len, f);
    fclose(f);
    printf("wrote %s  %zu bytes  (%.2f s of audio)\n", out, (size_t)wav_len,
           (double)(wav_len - 44) / 2 / 24000.0);
    printf("total %.1f s\n", now() - t0);

    voicevox_wav_free(wav);
    voicevox_synthesizer_delete(syn);
    voicevox_open_jtalk_rc_delete(ojt);
    printf("TTS OK\n");
    return 0;
}
