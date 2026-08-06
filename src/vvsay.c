// vvsay.c - say something, through whichever VOICEVOX CORE is linked.
//
// The smallest useful program that uses the API: load, synthesise, write a WAV.
// It is here as the worked example for the drop-in library - built against the
// official `voicevox_core.h` and linked against this project's
// `voicevox_core.dll`, it runs the official CORE inside the emulator without
// knowing that is what it is doing.
//
//     vvsay <onnxruntime> <dict-dir> <model.vvm> <text-file> [style] [out.wav]
//
// The text comes from a file rather than the command line on purpose: this
// project is developed on Windows through MSYS, where a UTF-8 argument has
// several chances to be mangled before main() sees it, and a file has none.
//
// Every step prints and flushes before it runs and its elapsed time after, so a
// run that takes hours under the emulator still says where it is.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "voicevox_core.h"

#if defined(_WIN32)
#include <windows.h>
static double now(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#else
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
#endif

static double t_step;

static void step(const char* what) {
    printf("step  %s\n", what);
    fflush(stdout);
    t_step = now();
}

static void done(void) {
    printf("      ... %.1f s\n", now() - t_step);
    fflush(stdout);
}

static void check(VoicevoxResultCode r, const char* what) {
    if (r == VOICEVOX_RESULT_OK) {
        done();
        return;
    }
    printf("FAIL  %s: %d %s\n", what, (int)r, voicevox_error_result_to_message(r));
    fflush(stdout);
    exit(1);
}

// Reads the whole file and trims one trailing newline, which every editor adds
// and which would otherwise become a pause at the end of the utterance.
static char* read_text(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    char* s = malloc((size_t)n + 1);
    if (!s) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(s, 1, (size_t)n, f);
    fclose(f);
    s[got] = 0;
    while (got && (s[got - 1] == '\n' || s[got - 1] == '\r')) s[--got] = 0;
    // A UTF-8 BOM would be part of the first mora otherwise.
    if (got >= 3 && (unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBB &&
        (unsigned char)s[2] == 0xBF)
        memmove(s, s + 3, got - 2);
    return s;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: vvsay <onnxruntime> <dict-dir> <model.vvm> <text-file> "
                "[style] [out.wav]\n");
        return 2;
    }
    const char* ort_path = argv[1];
    const char* dict = argv[2];
    const char* vvm = argv[3];
    const char* text_file = argv[4];
    uint32_t style = argc > 5 ? (uint32_t)atoi(argv[5]) : 3;
    const char* out = argc > 6 ? argv[6] : "out.wav";

    char* text = read_text(text_file);
    if (!text) {
        fprintf(stderr, "vvsay: cannot read %s\n", text_file);
        return 1;
    }

    double t0 = now();
    printf("core  %s\n", voicevox_get_version());
    printf("text  %s  (style %u)\n", text, style);
    fflush(stdout);

    step("voicevox_onnxruntime_load_once");
    VoicevoxLoadOnnxruntimeOptions oopts = voicevox_make_default_load_onnxruntime_options();
    oopts.filename = ort_path;
    const VoicevoxOnnxruntime* ort = NULL;
    check(voicevox_onnxruntime_load_once(oopts, &ort), "load_once");

    step("voicevox_open_jtalk_rc_new");
    OpenJtalkRc* ojt = NULL;
    check(voicevox_open_jtalk_rc_new(dict, &ojt), "open_jtalk_rc_new");

    step("voicevox_synthesizer_new");
    VoicevoxInitializeOptions iopts = voicevox_make_default_initialize_options();
    iopts.acceleration_mode = VOICEVOX_ACCELERATION_MODE_CPU;
    iopts.cpu_num_threads = 1;  // the emulator schedules guest threads cooperatively
    VoicevoxSynthesizer* syn = NULL;
    check(voicevox_synthesizer_new(ort, ojt, iopts, &syn), "synthesizer_new");

    step("voicevox_voice_model_file_open");
    VoicevoxVoiceModelFile* model = NULL;
    check(voicevox_voice_model_file_open(vvm, &model), "voice_model_file_open");

    step("voicevox_synthesizer_load_voice_model");
    check(voicevox_synthesizer_load_voice_model(syn, model), "load_voice_model");
    voicevox_voice_model_file_delete(model);

    step("voicevox_synthesizer_tts");
    VoicevoxTtsOptions topts = voicevox_make_default_tts_options();
    uintptr_t wav_len = 0;
    uint8_t* wav = NULL;
    check(voicevox_synthesizer_tts(syn, text, style, topts, &wav_len, &wav), "tts");

    FILE* f = fopen(out, "wb");
    if (!f) {
        printf("FAIL  cannot write %s\n", out);
        return 1;
    }
    fwrite(wav, 1, wav_len, f);
    fclose(f);
    printf("wrote %s  %zu bytes  (%.2f s of audio)\n", out, (size_t)wav_len,
           wav_len > 44 ? (double)(wav_len - 44) / 2 / 24000.0 : 0.0);
    printf("total %.1f s\n", now() - t0);

    voicevox_wav_free(wav);
    voicevox_synthesizer_delete(syn);
    voicevox_open_jtalk_rc_delete(ojt);
    free(text);
    printf("VVSAY OK\n");
    return 0;
}
