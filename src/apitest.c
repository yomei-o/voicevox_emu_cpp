// apitest.c - exercises the API surface, whichever implementation is linked.
//
// The claim this project makes is "the same API".  The way to check a claim
// like that is to call all of it and look at what comes back, so this walks
// the header from top to bottom: every entry point that can be reached without
// a second voice model gets called, its result code checked, and anything it
// produces printed in a form small enough to eyeball.
//
// It builds against the official libvoicevox_core just as happily as against
// this project's implementation, and the two are meant to print the same thing.
//
//     gcc -O2 -o apitest apitest.c -lvoicevox_core        # the original
//     gcc -O2 -o apitest apitest.c vvhost.c -lpthread     # this project
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "voicevox_core.h"

static int fails = 0;
static int checks = 0;

static void ok(const char* what, VoicevoxResultCode r) {
    checks++;
    if (r == VOICEVOX_RESULT_OK) {
        printf("ok    %s\n", what);
    } else {
        printf("FAIL  %s -> %d %s\n", what, (int)r, voicevox_error_result_to_message(r));
        fails++;
    }
}

// Some calls are expected to fail - a style that is not in this model, a word
// uuid that has been removed - and the code they return is part of the API.
static void expect(const char* what, VoicevoxResultCode got, VoicevoxResultCode want) {
    checks++;
    if (got == want) {
        printf("ok    %s -> %d as expected\n", what, (int)got);
    } else {
        printf("FAIL  %s -> %d (%s), expected %d\n", what, (int)got,
               voicevox_error_result_to_message(got), (int)want);
        fails++;
    }
}

static void note(const char* what, const char* value) {
    printf("      %-34s %s\n", what, value ? value : "(null)");
}

// JSON here runs to tens of kilobytes; the shape is what matters.
static void note_json(const char* what, const char* json) {
    char buf[121];
    if (!json) {
        note(what, "(null)");
        return;
    }
    size_t n = strlen(json);
    snprintf(buf, sizeof buf, "%.100s", json);
    printf("      %-34s %zu bytes: %s%s\n", what, n, buf, n > 100 ? "..." : "");
}

static void note_uuid(const char* what, const uint8_t id[16]) {
    char buf[40];
    for (int i = 0; i < 16; i++) snprintf(buf + i * 2, 3, "%02x", id[i]);
    note(what, buf);
}

static int write_file(const char* path, const void* data, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    size_t put = fwrite(data, 1, n, f);
    fclose(f);
    return put == n;
}

int main(int argc, char** argv) {
    const char* ort_path = argc > 1 ? argv[1] : "/opt/vv/libvoicevox_onnxruntime.so.1.17.3";
    const char* dict = argc > 2 ? argv[2] : "/opt/vv/open_jtalk_dic_utf_8-1.11";
    const char* vvm = argc > 3 ? argv[3] : "/opt/vv/0.vvm";
    const char* outdir = argc > 4 ? argv[4] : "/opt/vv";
    uint32_t style = argc > 5 ? (uint32_t)atoi(argv[5]) : 3;

    printf("== library\n");
    note("voicevox_get_version", voicevox_get_version());
    note("onnxruntime lib, versioned", voicevox_get_onnxruntime_lib_versioned_filename());
    note("onnxruntime lib, unversioned", voicevox_get_onnxruntime_lib_unversioned_filename());
    note("error_result_to_message(0)", voicevox_error_result_to_message(VOICEVOX_RESULT_OK));
    note("error_result_to_message(6)",
         voicevox_error_result_to_message(VOICEVOX_RESULT_STYLE_NOT_FOUND_ERROR));

    printf("\n== defaults\n");
    {
        VoicevoxLoadOnnxruntimeOptions lo = voicevox_make_default_load_onnxruntime_options();
        note("default onnxruntime filename", lo.filename);
        VoicevoxInitializeOptions io = voicevox_make_default_initialize_options();
        char buf[64];
        snprintf(buf, sizeof buf, "acceleration=%d threads=%u", (int)io.acceleration_mode,
                 (unsigned)io.cpu_num_threads);
        note("default initialize options", buf);
        VoicevoxSynthesisOptions so = voicevox_make_default_synthesis_options();
        VoicevoxTtsOptions to = voicevox_make_default_tts_options();
        snprintf(buf, sizeof buf, "synthesis=%d tts=%d", (int)so.enable_interrogative_upspeak,
                 (int)to.enable_interrogative_upspeak);
        note("default upspeak", buf);
    }

    printf("\n== onnxruntime\n");
    VoicevoxLoadOnnxruntimeOptions oopts = voicevox_make_default_load_onnxruntime_options();
    oopts.filename = ort_path;
    const VoicevoxOnnxruntime* ort = NULL;
    ok("voicevox_onnxruntime_load_once", voicevox_onnxruntime_load_once(oopts, &ort));
    checks++;
    if (voicevox_onnxruntime_get() == ort) {
        printf("ok    voicevox_onnxruntime_get returns the same singleton\n");
    } else {
        printf("FAIL  voicevox_onnxruntime_get disagrees with load_once\n");
        fails++;
    }
    {
        char* json = NULL;
        ok("voicevox_onnxruntime_create_supported_devices_json",
           voicevox_onnxruntime_create_supported_devices_json(ort, &json));
        note_json("supported devices", json);
        voicevox_json_free(json);
    }

    printf("\n== open jtalk\n");
    OpenJtalkRc* ojt = NULL;
    ok("voicevox_open_jtalk_rc_new", voicevox_open_jtalk_rc_new(dict, &ojt));
    {
        char* json = NULL;
        ok("voicevox_open_jtalk_rc_analyze",
           voicevox_open_jtalk_rc_analyze(ojt, "こんにちはなのだ", &json));
        note_json("accent phrases from analyze", json);
        // The validator wants one accent phrase, and analyze returns an array
        // of them, so cut the first element out by matching its braces.
        if (json && json[0] == '[') {
            int depth = 0;
            size_t end = 0;
            for (size_t i = 1; json[i]; i++) {
                if (json[i] == '{') depth++;
                if (json[i] == '}' && --depth == 0) {
                    end = i + 1;
                    break;
                }
            }
            if (end > 1) {
                char* one = malloc(end);
                memcpy(one, json + 1, end - 1);
                one[end - 1] = 0;
                ok("voicevox_accent_phrase_validate on its first element",
                   voicevox_accent_phrase_validate(one));
                free(one);
            }
        }
        voicevox_json_free(json);
    }

    printf("\n== user dictionary\n");
    VoicevoxUserDict* dict1 = voicevox_user_dict_new();
    checks++;
    if (dict1) {
        printf("ok    voicevox_user_dict_new\n");
    } else {
        printf("FAIL  voicevox_user_dict_new returned NULL\n");
        fails++;
    }
    if (dict1) {
        VoicevoxUserDictWord w = voicevox_user_dict_word_make("虚無", "キョム", 1);
        uint8_t uuid[16];
        ok("voicevox_user_dict_add_word", voicevox_user_dict_add_word(dict1, &w, &uuid));
        note_uuid("word uuid", uuid);
        char* json = NULL;
        ok("voicevox_user_dict_to_json", voicevox_user_dict_to_json(dict1, &json));
        note_json("user dict", json);
        voicevox_json_free(json);

        w.priority = 8;
        ok("voicevox_user_dict_update_word", voicevox_user_dict_update_word(dict1, &uuid, &w));

        char path[1024];
        snprintf(path, sizeof path, "%s/apitest_dict.json", outdir);
        ok("voicevox_user_dict_save", voicevox_user_dict_save(dict1, path));
        VoicevoxUserDict* dict2 = voicevox_user_dict_new();
        ok("voicevox_user_dict_load", voicevox_user_dict_load(dict2, path));
        ok("voicevox_user_dict_import", voicevox_user_dict_import(dict1, dict2));
        ok("voicevox_open_jtalk_rc_use_user_dict",
           voicevox_open_jtalk_rc_use_user_dict(ojt, dict1));
        ok("voicevox_user_dict_remove_word", voicevox_user_dict_remove_word(dict1, &uuid));
        expect("voicevox_user_dict_remove_word again",
               voicevox_user_dict_remove_word(dict1, &uuid),
               VOICEVOX_RESULT_USER_DICT_WORD_NOT_FOUND_ERROR);
        voicevox_user_dict_delete(dict2);
    }

    printf("\n== voice model file\n");
    VoicevoxVoiceModelFile* model = NULL;
    ok("voicevox_voice_model_file_open", voicevox_voice_model_file_open(vvm, &model));
    uint8_t model_id[16];
    voicevox_voice_model_file_id(model, &model_id);
    note_uuid("voicevox_voice_model_file_id", model_id);
    {
        char* json = voicevox_voice_model_file_create_metas_json(model);
        note_json("model metas", json);
        voicevox_json_free(json);
    }

    printf("\n== synthesizer\n");
    VoicevoxInitializeOptions iopts = voicevox_make_default_initialize_options();
    iopts.acceleration_mode = VOICEVOX_ACCELERATION_MODE_CPU;
    iopts.cpu_num_threads = 1;
    VoicevoxSynthesizer* syn = NULL;
    ok("voicevox_synthesizer_new", voicevox_synthesizer_new(ort, ojt, iopts, &syn));
    checks++;
    if (voicevox_synthesizer_get_onnxruntime(syn) == ort) {
        printf("ok    voicevox_synthesizer_get_onnxruntime matches\n");
    } else {
        printf("FAIL  voicevox_synthesizer_get_onnxruntime does not match\n");
        fails++;
    }
    printf("      %-34s %s\n", "voicevox_synthesizer_is_gpu_mode",
           voicevox_synthesizer_is_gpu_mode(syn) ? "true" : "false");
    checks++;
    if (!voicevox_synthesizer_is_loaded_voice_model(syn, &model_id)) {
        printf("ok    is_loaded_voice_model is false before loading\n");
    } else {
        printf("FAIL  is_loaded_voice_model is true before loading\n");
        fails++;
    }

    ok("voicevox_synthesizer_load_voice_model",
       voicevox_synthesizer_load_voice_model(syn, model));
    checks++;
    if (voicevox_synthesizer_is_loaded_voice_model(syn, &model_id)) {
        printf("ok    is_loaded_voice_model is true after loading\n");
    } else {
        printf("FAIL  is_loaded_voice_model is false after loading\n");
        fails++;
    }
    expect("loading the same model twice",
           voicevox_synthesizer_load_voice_model(syn, model),
           VOICEVOX_RESULT_MODEL_ALREADY_LOADED_ERROR);
    voicevox_voice_model_file_delete(model);
    {
        char* json = voicevox_synthesizer_create_metas_json(syn);
        note_json("synthesizer metas", json);
        voicevox_json_free(json);
    }

    printf("\n== text to query to audio\n");
    char* query = NULL;
    ok("voicevox_synthesizer_create_audio_query",
       voicevox_synthesizer_create_audio_query(syn, "ずんだもんなのだ", style, &query));
    note_json("audio query", query);
    ok("voicevox_audio_query_validate", voicevox_audio_query_validate(query));

    {
        char* phrases = NULL;
        ok("voicevox_synthesizer_create_accent_phrases",
           voicevox_synthesizer_create_accent_phrases(syn, "ずんだもんなのだ", style, &phrases));
        char* replaced = NULL;
        ok("voicevox_synthesizer_replace_mora_data",
           voicevox_synthesizer_replace_mora_data(syn, phrases, style, &replaced));
        voicevox_json_free(replaced);
        replaced = NULL;
        ok("voicevox_synthesizer_replace_phoneme_length",
           voicevox_synthesizer_replace_phoneme_length(syn, phrases, style, &replaced));
        voicevox_json_free(replaced);
        replaced = NULL;
        ok("voicevox_synthesizer_replace_mora_pitch",
           voicevox_synthesizer_replace_mora_pitch(syn, phrases, style, &replaced));
        voicevox_json_free(replaced);

        char* fromphrases = NULL;
        ok("voicevox_audio_query_create_from_accent_phrases",
           voicevox_audio_query_create_from_accent_phrases(phrases, &fromphrases));
        note_json("audio query from accent phrases", fromphrases);
        voicevox_json_free(fromphrases);
        voicevox_json_free(phrases);
    }

    {
        char* kana_query = NULL;
        ok("voicevox_synthesizer_create_audio_query_from_kana",
           voicevox_synthesizer_create_audio_query_from_kana(syn, "コンニチワ'", style,
                                                             &kana_query));
        voicevox_json_free(kana_query);
        char* kana_phrases = NULL;
        ok("voicevox_synthesizer_create_accent_phrases_from_kana",
           voicevox_synthesizer_create_accent_phrases_from_kana(syn, "コンニチワ'", style,
                                                                &kana_phrases));
        voicevox_json_free(kana_phrases);
    }

    {
        VoicevoxSynthesisOptions sopts = voicevox_make_default_synthesis_options();
        uintptr_t len = 0;
        uint8_t* wav = NULL;
        ok("voicevox_synthesizer_synthesis",
           voicevox_synthesizer_synthesis(syn, query, style, sopts, &len, &wav));
        char path[1024], buf[64];
        snprintf(path, sizeof path, "%s/apitest_synthesis.wav", outdir);
        snprintf(buf, sizeof buf, "%zu bytes (%.2f s)", (size_t)len,
                 len > 44 ? (double)(len - 44) / 2 / 24000.0 : 0.0);
        note("synthesis wav", buf);
        if (!write_file(path, wav, len)) printf("      (could not write %s)\n", path);
        voicevox_wav_free(wav);
    }
    voicevox_json_free(query);

    {
        VoicevoxTtsOptions topts = voicevox_make_default_tts_options();
        uintptr_t len = 0;
        uint8_t* wav = NULL;
        ok("voicevox_synthesizer_tts",
           voicevox_synthesizer_tts(syn, "ずんだもんなのだ", style, topts, &len, &wav));
        char path[1024], buf[64];
        snprintf(path, sizeof path, "%s/apitest_tts.wav", outdir);
        snprintf(buf, sizeof buf, "%zu bytes (%.2f s)", (size_t)len,
                 len > 44 ? (double)(len - 44) / 2 / 24000.0 : 0.0);
        note("tts wav", buf);
        if (!write_file(path, wav, len)) printf("      (could not write %s)\n", path);
        voicevox_wav_free(wav);

        len = 0;
        wav = NULL;
        ok("voicevox_synthesizer_tts_from_kana",
           voicevox_synthesizer_tts_from_kana(syn, "ズンダモンナノダ'", style, topts, &len, &wav));
        snprintf(buf, sizeof buf, "%zu bytes", (size_t)len);
        note("tts_from_kana wav", buf);
        voicevox_wav_free(wav);
    }

    printf("\n== error paths\n");
    {
        char* json = NULL;
        expect("create_audio_query with an unknown style",
               voicevox_synthesizer_create_audio_query(syn, "テスト", 99999, &json),
               VOICEVOX_RESULT_STYLE_NOT_FOUND_ERROR);
        voicevox_json_free(json);
        expect("audio_query_validate on nonsense",
               voicevox_audio_query_validate("{\"not\":\"a query\"}"),
               VOICEVOX_RESULT_INVALID_AUDIO_QUERY_ERROR);
        expect("mora_validate on nonsense", voicevox_mora_validate("{}"),
               VOICEVOX_RESULT_INVALID_MORA_ERROR);
        expect("score_validate on nonsense", voicevox_score_validate("{}"),
               VOICEVOX_RESULT_INVALID_SCORE_ERROR);
        expect("note_validate on nonsense", voicevox_note_validate("{}"),
               VOICEVOX_RESULT_INVALID_NOTE_ERROR);
        expect("frame_audio_query_validate on nonsense", voicevox_frame_audio_query_validate("{}"),
               VOICEVOX_RESULT_INVALID_FRAME_AUDIO_QUERY_ERROR);
        expect("frame_phoneme_validate on nonsense", voicevox_frame_phoneme_validate("{}"),
               VOICEVOX_RESULT_INVALID_FRAME_PHONEME_ERROR);
        expect("create_audio_query_from_kana on bad kana",
               voicevox_synthesizer_create_audio_query_from_kana(syn, "これはカナではない", style,
                                                                 &json),
               VOICEVOX_RESULT_PARSE_KANA_ERROR);
        voicevox_json_free(json);
    }

    printf("\n== singing\n");
    // Whether this model sings is a property of the model, not of the API; the
    // result code is what is being checked, either way.
    {
        const char* score =
            "{\"notes\":[{\"id\":null,\"key\":null,\"frame_length\":15,\"lyric\":\"\"},"
            "{\"id\":null,\"key\":60,\"frame_length\":45,\"lyric\":\"ド\"},"
            "{\"id\":null,\"key\":null,\"frame_length\":15,\"lyric\":\"\"}]}";
        printf("      %-34s %d\n", "voicevox_score_validate", (int)voicevox_score_validate(score));
        char* fq = NULL;
        VoicevoxResultCode r =
            voicevox_synthesizer_create_sing_frame_audio_query(syn, score, style, &fq);
        printf("      %-34s %d %s\n", "create_sing_frame_audio_query", (int)r,
               voicevox_error_result_to_message(r));
        if (r == VOICEVOX_RESULT_OK) {
            note_json("frame audio query", fq);
            char* f0 = NULL;
            ok("voicevox_synthesizer_create_sing_frame_f0",
               voicevox_synthesizer_create_sing_frame_f0(syn, score, fq, style, &f0));
            voicevox_json_free(f0);
            char* vol = NULL;
            ok("voicevox_synthesizer_create_sing_frame_volume",
               voicevox_synthesizer_create_sing_frame_volume(syn, score, fq, style, &vol));
            voicevox_json_free(vol);
            ok("voicevox_ensure_compatible", voicevox_ensure_compatible(score, fq));
            uintptr_t len = 0;
            uint8_t* wav = NULL;
            ok("voicevox_synthesizer_frame_synthesis",
               voicevox_synthesizer_frame_synthesis(syn, fq, style, &len, &wav));
            voicevox_wav_free(wav);
        }
        voicevox_json_free(fq);
    }

    printf("\n== teardown\n");
    ok("voicevox_synthesizer_unload_voice_model",
       voicevox_synthesizer_unload_voice_model(syn, &model_id));
    checks++;
    if (!voicevox_synthesizer_is_loaded_voice_model(syn, &model_id)) {
        printf("ok    is_loaded_voice_model is false after unloading\n");
    } else {
        printf("FAIL  is_loaded_voice_model is true after unloading\n");
        fails++;
    }
    voicevox_synthesizer_delete(syn);
    voicevox_open_jtalk_rc_delete(ojt);
    if (dict1) voicevox_user_dict_delete(dict1);

    printf("\n%d checks, %d failed\n", checks, fails);
    printf("%s\n", fails ? "APITEST FAILED" : "APITEST OK");
    return fails ? 1 : 0;
}
