// vvagent.c - the guest half of the API.
//
// Runs inside the emulator with the official libvoicevox_core.so linked in,
// reads one request from stdin, makes the call, writes one response to stdout,
// repeats.  Nothing is decompiled and nothing is extracted: this is the library
// being used through its own published interface, with a pipe where the caller
// would normally be.
//
// stdout is the protocol and nothing else.  The core logs to stderr, and so
// does the emulator, so both stay out of the way.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "voicevox_core.h"
#include "vvrpc.h"

// ---- framing --------------------------------------------------------------

static int read_all(void* p, size_t n) {
    unsigned char* b = p;
    while (n) {
        ssize_t got = read(0, b, n);
        if (got == 0) return 0;  // the host closed the pipe: time to stop
        if (got < 0) return -1;
        b += got;
        n -= (size_t)got;
    }
    return 1;
}

static int write_all(const void* p, size_t n) {
    const unsigned char* b = p;
    while (n) {
        ssize_t put = write(1, b, n);
        if (put <= 0) return -1;
        b += put;
        n -= (size_t)put;
    }
    return 0;
}

// Everything a request owns, freed together at the end of the iteration.
typedef struct {
    uint32_t op;
    uint32_t nblob;
    uint64_t s[VV_MAX_SCALARS];
    uint32_t len[VV_MAX_BLOBS];
    char* blob[VV_MAX_BLOBS];  // always NUL-terminated, so they can be used as strings
} Req;

static int read_request(Req* q) {
    uint32_t head[2];
    int r = read_all(head, sizeof head);
    if (r <= 0) return r;
    q->op = head[0];
    q->nblob = head[1];
    if (q->nblob > VV_MAX_BLOBS) return -1;
    if (read_all(q->s, sizeof q->s) <= 0) return -1;
    memset(q->blob, 0, sizeof q->blob);
    for (uint32_t i = 0; i < q->nblob; i++) {
        if (read_all(&q->len[i], 4) <= 0) return -1;
        if (q->len[i] > VV_MAX_BLOB_BYTES) return -1;
        q->blob[i] = malloc((size_t)q->len[i] + 1);
        if (!q->blob[i]) return -1;
        if (q->len[i] && read_all(q->blob[i], q->len[i]) <= 0) return -1;
        q->blob[i][q->len[i]] = 0;
    }
    return 1;
}

static void free_request(Req* q) {
    for (int i = 0; i < VV_MAX_BLOBS; i++) {
        free(q->blob[i]);
        q->blob[i] = NULL;
    }
}

// The reply is assembled in one of these and sent by send().
typedef struct {
    int32_t code;
    uint32_t nblob;
    uint64_t r[2];
    uint32_t len[VV_MAX_BLOBS];
    const void* blob[VV_MAX_BLOBS];
} Res;

static int send_response(const Res* p) {
    uint32_t head[2];
    memcpy(&head[0], &p->code, 4);
    head[1] = p->nblob;
    if (write_all(head, sizeof head) < 0) return -1;
    if (write_all(p->r, sizeof p->r) < 0) return -1;
    for (uint32_t i = 0; i < p->nblob; i++) {
        if (write_all(&p->len[i], 4) < 0) return -1;
        if (p->len[i] && write_all(p->blob[i], p->len[i]) < 0) return -1;
    }
    // The host is waiting on this and nothing else will push it out.
    fflush(stdout);
    return 0;
}

// ---- shorthands used by the dispatch --------------------------------------

#define OJT(x) ((struct OpenJtalkRc*)(uintptr_t)(x))
#define ORT(x) ((const struct VoicevoxOnnxruntime*)(uintptr_t)(x))
#define SYN(x) ((const struct VoicevoxSynthesizer*)(uintptr_t)(x))
#define MODEL(x) ((const struct VoicevoxVoiceModelFile*)(uintptr_t)(x))
#define UDICT(x) ((const struct VoicevoxUserDict*)(uintptr_t)(x))
#define H(p) ((uint64_t)(uintptr_t)(p))

// A JSON answer: hand it over, then give it back to the core.  The host gets a
// copy in its own heap and frees that with its own voicevox_json_free, so no
// allocation ever has to survive a round trip.
static void reply_json_owned(Res* res, char* json) {
    static char* held;
    if (held) voicevox_json_free(held);
    held = json;
    res->nblob = 1;
    res->blob[0] = json ? json : "";
    res->len[0] = json ? (uint32_t)strlen(json) : 0;
}

static void reply_wav_owned(Res* res, uint8_t* wav, uintptr_t len) {
    static uint8_t* held;
    if (held) voicevox_wav_free(held);
    held = wav;
    res->nblob = 1;
    res->blob[0] = wav ? (const void*)wav : (const void*)"";
    res->len[0] = wav ? (uint32_t)len : 0;
}

static void reply_str(Res* res, const char* s) {
    res->nblob = 1;
    res->blob[0] = s ? s : "";
    res->len[0] = s ? (uint32_t)strlen(s) : 0;
}

// ---- dispatch -------------------------------------------------------------

static void dispatch(const Req* q, Res* res) {
    switch (q->op) {
        case VV_OP_PING:
            res->r[0] = 0x76765250;  // "vvRP"
            break;

        case VV_OP_ORT_LIB_VERSIONED_FILENAME:
            reply_str(res, voicevox_get_onnxruntime_lib_versioned_filename());
            break;
        case VV_OP_ORT_LIB_UNVERSIONED_FILENAME:
            reply_str(res, voicevox_get_onnxruntime_lib_unversioned_filename());
            break;
        case VV_OP_GET_VERSION:
            reply_str(res, voicevox_get_version());
            break;

        case VV_OP_DEFAULT_LOAD_ORT_OPTIONS: {
            VoicevoxLoadOnnxruntimeOptions o = voicevox_make_default_load_onnxruntime_options();
            reply_str(res, o.filename);
            break;
        }
        case VV_OP_DEFAULT_INITIALIZE_OPTIONS: {
            VoicevoxInitializeOptions o = voicevox_make_default_initialize_options();
            res->r[0] = (uint64_t)(uint32_t)o.acceleration_mode;
            res->r[1] = o.cpu_num_threads;
            break;
        }
        case VV_OP_DEFAULT_SYNTHESIS_OPTIONS: {
            VoicevoxSynthesisOptions o = voicevox_make_default_synthesis_options();
            res->r[0] = o.enable_interrogative_upspeak ? 1 : 0;
            break;
        }
        case VV_OP_DEFAULT_TTS_OPTIONS: {
            VoicevoxTtsOptions o = voicevox_make_default_tts_options();
            res->r[0] = o.enable_interrogative_upspeak ? 1 : 0;
            break;
        }
        case VV_OP_DEFAULT_USER_DICT_WORD: {
            VoicevoxUserDictWord w = voicevox_user_dict_word_make("あ", "ア", 0);
            res->r[0] = (uint64_t)(uint32_t)w.word_type;
            res->r[1] = w.priority;
            break;
        }

        case VV_OP_ORT_GET:
            res->r[0] = H(voicevox_onnxruntime_get());
            break;
        case VV_OP_ORT_LOAD_ONCE: {
            VoicevoxLoadOnnxruntimeOptions o = voicevox_make_default_load_onnxruntime_options();
            if (q->nblob > 0 && q->len[0]) o.filename = q->blob[0];
            const struct VoicevoxOnnxruntime* ort = NULL;
            res->code = voicevox_onnxruntime_load_once(o, &ort);
            res->r[0] = H(ort);
            break;
        }
        case VV_OP_ORT_INIT_ONCE: {
#if defined(VOICEVOX_LINK_ONNXRUNTIME)
            const struct VoicevoxOnnxruntime* ort = NULL;
            res->code = voicevox_onnxruntime_init_once(&ort);
            res->r[0] = H(ort);
#else
            // This build of CORE loads ONNX Runtime rather than linking it, so
            // the header does not declare init_once at all.
            res->code = VOICEVOX_RESULT_INIT_INFERENCE_RUNTIME_ERROR;
#endif
            break;
        }
        case VV_OP_ORT_SUPPORTED_DEVICES: {
            char* json = NULL;
            res->code = voicevox_onnxruntime_create_supported_devices_json(ORT(q->s[0]), &json);
            reply_json_owned(res, json);
            break;
        }

        case VV_OP_OPEN_JTALK_NEW: {
            struct OpenJtalkRc* ojt = NULL;
            res->code = voicevox_open_jtalk_rc_new(q->blob[0], &ojt);
            res->r[0] = H(ojt);
            break;
        }
        case VV_OP_OPEN_JTALK_USE_USER_DICT:
            res->code = voicevox_open_jtalk_rc_use_user_dict(OJT(q->s[0]), UDICT(q->s[1]));
            break;
        case VV_OP_OPEN_JTALK_ANALYZE: {
            char* json = NULL;
            res->code = voicevox_open_jtalk_rc_analyze(OJT(q->s[0]), q->blob[0], &json);
            reply_json_owned(res, json);
            break;
        }
        case VV_OP_OPEN_JTALK_DELETE:
            voicevox_open_jtalk_rc_delete(OJT(q->s[0]));
            break;

        case VV_OP_AUDIO_QUERY_FROM_ACCENT_PHRASES: {
            char* json = NULL;
            res->code = voicevox_audio_query_create_from_accent_phrases(q->blob[0], &json);
            reply_json_owned(res, json);
            break;
        }
        case VV_OP_AUDIO_QUERY_VALIDATE:
            res->code = voicevox_audio_query_validate(q->blob[0]);
            break;
        case VV_OP_ACCENT_PHRASE_VALIDATE:
            res->code = voicevox_accent_phrase_validate(q->blob[0]);
            break;
        case VV_OP_MORA_VALIDATE:
            res->code = voicevox_mora_validate(q->blob[0]);
            break;
        case VV_OP_SCORE_VALIDATE:
            res->code = voicevox_score_validate(q->blob[0]);
            break;
        case VV_OP_NOTE_VALIDATE:
            res->code = voicevox_note_validate(q->blob[0]);
            break;
        case VV_OP_FRAME_AUDIO_QUERY_VALIDATE:
            res->code = voicevox_frame_audio_query_validate(q->blob[0]);
            break;
        case VV_OP_FRAME_PHONEME_VALIDATE:
            res->code = voicevox_frame_phoneme_validate(q->blob[0]);
            break;
        case VV_OP_ENSURE_COMPATIBLE:
            res->code = voicevox_ensure_compatible(q->blob[0], q->blob[1]);
            break;

        case VV_OP_MODEL_OPEN: {
            struct VoicevoxVoiceModelFile* m = NULL;
            res->code = voicevox_voice_model_file_open(q->blob[0], &m);
            res->r[0] = H(m);
            break;
        }
        case VV_OP_MODEL_ID: {
            static uint8_t id[16];
            voicevox_voice_model_file_id(MODEL(q->s[0]), &id);
            res->nblob = 1;
            res->blob[0] = id;
            res->len[0] = 16;
            break;
        }
        case VV_OP_MODEL_METAS_JSON:
            reply_json_owned(res, voicevox_voice_model_file_create_metas_json(MODEL(q->s[0])));
            break;
        case VV_OP_MODEL_DELETE:
            voicevox_voice_model_file_delete((struct VoicevoxVoiceModelFile*)MODEL(q->s[0]));
            break;

        case VV_OP_SYN_NEW: {
            VoicevoxInitializeOptions o = voicevox_make_default_initialize_options();
            o.acceleration_mode = (VoicevoxAccelerationMode)(int32_t)(uint32_t)q->s[2];
            o.cpu_num_threads = (uint16_t)q->s[3];
            struct VoicevoxSynthesizer* syn = NULL;
            res->code = voicevox_synthesizer_new(ORT(q->s[0]), OJT(q->s[1]), o, &syn);
            res->r[0] = H(syn);
            break;
        }
        case VV_OP_SYN_DELETE:
            voicevox_synthesizer_delete((struct VoicevoxSynthesizer*)SYN(q->s[0]));
            break;
        case VV_OP_SYN_LOAD_VOICE_MODEL:
            res->code = voicevox_synthesizer_load_voice_model(SYN(q->s[0]), MODEL(q->s[1]));
            break;
        case VV_OP_SYN_UNLOAD_VOICE_MODEL:
            res->code = voicevox_synthesizer_unload_voice_model(
                SYN(q->s[0]), (VoicevoxVoiceModelId)q->blob[0]);
            break;
        case VV_OP_SYN_GET_ORT:
            res->r[0] = H(voicevox_synthesizer_get_onnxruntime(SYN(q->s[0])));
            break;
        case VV_OP_SYN_IS_GPU_MODE:
            res->r[0] = voicevox_synthesizer_is_gpu_mode(SYN(q->s[0])) ? 1 : 0;
            break;
        case VV_OP_SYN_IS_LOADED_VOICE_MODEL:
            res->r[0] = voicevox_synthesizer_is_loaded_voice_model(
                            SYN(q->s[0]), (VoicevoxVoiceModelId)q->blob[0])
                            ? 1
                            : 0;
            break;
        case VV_OP_SYN_METAS_JSON:
            reply_json_owned(res, voicevox_synthesizer_create_metas_json(SYN(q->s[0])));
            break;

        case VV_OP_SYN_AUDIO_QUERY_FROM_KANA:
        case VV_OP_SYN_AUDIO_QUERY:
        case VV_OP_SYN_ACCENT_PHRASES_FROM_KANA:
        case VV_OP_SYN_ACCENT_PHRASES:
        case VV_OP_SYN_REPLACE_MORA_DATA:
        case VV_OP_SYN_REPLACE_PHONEME_LENGTH:
        case VV_OP_SYN_REPLACE_MORA_PITCH: {
            // All seven are (synthesizer, one string, style id) -> JSON.
            const struct VoicevoxSynthesizer* syn = SYN(q->s[0]);
            VoicevoxStyleId style = (VoicevoxStyleId)q->s[1];
            const char* in = q->blob[0];
            char* json = NULL;
            switch (q->op) {
                case VV_OP_SYN_AUDIO_QUERY_FROM_KANA:
                    res->code = voicevox_synthesizer_create_audio_query_from_kana(syn, in, style,
                                                                                  &json);
                    break;
                case VV_OP_SYN_AUDIO_QUERY:
                    res->code = voicevox_synthesizer_create_audio_query(syn, in, style, &json);
                    break;
                case VV_OP_SYN_ACCENT_PHRASES_FROM_KANA:
                    res->code = voicevox_synthesizer_create_accent_phrases_from_kana(syn, in, style,
                                                                                     &json);
                    break;
                case VV_OP_SYN_ACCENT_PHRASES:
                    res->code = voicevox_synthesizer_create_accent_phrases(syn, in, style, &json);
                    break;
                case VV_OP_SYN_REPLACE_MORA_DATA:
                    res->code = voicevox_synthesizer_replace_mora_data(syn, in, style, &json);
                    break;
                case VV_OP_SYN_REPLACE_PHONEME_LENGTH:
                    res->code = voicevox_synthesizer_replace_phoneme_length(syn, in, style, &json);
                    break;
                default:
                    res->code = voicevox_synthesizer_replace_mora_pitch(syn, in, style, &json);
                    break;
            }
            reply_json_owned(res, json);
            break;
        }

        case VV_OP_SYN_SYNTHESIS: {
            VoicevoxSynthesisOptions o = voicevox_make_default_synthesis_options();
            o.enable_interrogative_upspeak = q->s[2] != 0;
            uintptr_t len = 0;
            uint8_t* wav = NULL;
            res->code = voicevox_synthesizer_synthesis(SYN(q->s[0]), q->blob[0],
                                                       (VoicevoxStyleId)q->s[1], o, &len, &wav);
            reply_wav_owned(res, wav, len);
            break;
        }
        case VV_OP_SYN_TTS_FROM_KANA:
        case VV_OP_SYN_TTS: {
            VoicevoxTtsOptions o = voicevox_make_default_tts_options();
            o.enable_interrogative_upspeak = q->s[2] != 0;
            uintptr_t len = 0;
            uint8_t* wav = NULL;
            res->code = q->op == VV_OP_SYN_TTS
                            ? voicevox_synthesizer_tts(SYN(q->s[0]), q->blob[0],
                                                       (VoicevoxStyleId)q->s[1], o, &len, &wav)
                            : voicevox_synthesizer_tts_from_kana(SYN(q->s[0]), q->blob[0],
                                                                 (VoicevoxStyleId)q->s[1], o, &len,
                                                                 &wav);
            reply_wav_owned(res, wav, len);
            break;
        }
        case VV_OP_SYN_FRAME_SYNTHESIS: {
            uintptr_t len = 0;
            uint8_t* wav = NULL;
            res->code = voicevox_synthesizer_frame_synthesis(
                SYN(q->s[0]), q->blob[0], (VoicevoxStyleId)q->s[1], &len, &wav);
            reply_wav_owned(res, wav, len);
            break;
        }

        case VV_OP_SYN_SING_FRAME_AUDIO_QUERY: {
            char* json = NULL;
            res->code = voicevox_synthesizer_create_sing_frame_audio_query(
                SYN(q->s[0]), q->blob[0], (VoicevoxStyleId)q->s[1], &json);
            reply_json_owned(res, json);
            break;
        }
        case VV_OP_SYN_SING_FRAME_F0:
        case VV_OP_SYN_SING_FRAME_VOLUME: {
            char* json = NULL;
            res->code = q->op == VV_OP_SYN_SING_FRAME_F0
                            ? voicevox_synthesizer_create_sing_frame_f0(
                                  SYN(q->s[0]), q->blob[0], q->blob[1],
                                  (VoicevoxStyleId)q->s[1], &json)
                            : voicevox_synthesizer_create_sing_frame_volume(
                                  SYN(q->s[0]), q->blob[0], q->blob[1],
                                  (VoicevoxStyleId)q->s[1], &json);
            reply_json_owned(res, json);
            break;
        }

        case VV_OP_USER_DICT_NEW:
            res->r[0] = H(voicevox_user_dict_new());
            break;
        case VV_OP_USER_DICT_LOAD:
            res->code = voicevox_user_dict_load(UDICT(q->s[0]), q->blob[0]);
            break;
        case VV_OP_USER_DICT_ADD_WORD: {
            VoicevoxUserDictWord w = voicevox_user_dict_word_make(q->blob[0], q->blob[1],
                                                                 (uintptr_t)q->s[1]);
            w.word_type = (VoicevoxUserDictWordType)(int32_t)(uint32_t)q->s[2];
            w.priority = (uint32_t)q->s[3];
            static uint8_t uuid[16];
            res->code = voicevox_user_dict_add_word(UDICT(q->s[0]), &w, &uuid);
            res->nblob = 1;
            res->blob[0] = uuid;
            res->len[0] = 16;
            break;
        }
        case VV_OP_USER_DICT_UPDATE_WORD: {
            VoicevoxUserDictWord w = voicevox_user_dict_word_make(q->blob[0], q->blob[1],
                                                                 (uintptr_t)q->s[1]);
            w.word_type = (VoicevoxUserDictWordType)(int32_t)(uint32_t)q->s[2];
            w.priority = (uint32_t)q->s[3];
            res->code = voicevox_user_dict_update_word(
                UDICT(q->s[0]), (const uint8_t(*)[16])q->blob[2], &w);
            break;
        }
        case VV_OP_USER_DICT_REMOVE_WORD:
            res->code = voicevox_user_dict_remove_word(UDICT(q->s[0]),
                                                       (const uint8_t(*)[16])q->blob[0]);
            break;
        case VV_OP_USER_DICT_TO_JSON: {
            char* json = NULL;
            res->code = voicevox_user_dict_to_json(UDICT(q->s[0]), &json);
            reply_json_owned(res, json);
            break;
        }
        case VV_OP_USER_DICT_IMPORT:
            res->code = voicevox_user_dict_import(UDICT(q->s[0]), UDICT(q->s[1]));
            break;
        case VV_OP_USER_DICT_SAVE:
            res->code = voicevox_user_dict_save(UDICT(q->s[0]), q->blob[0]);
            break;
        case VV_OP_USER_DICT_DELETE:
            voicevox_user_dict_delete((struct VoicevoxUserDict*)UDICT(q->s[0]));
            break;

        case VV_OP_ERROR_MESSAGE:
            reply_str(res, voicevox_error_result_to_message((VoicevoxResultCode)(int32_t)(uint32_t)q->s[0]));
            break;

        default:
            // An opcode this build does not know is a version mismatch between
            // the two halves, and saying so beats a hang.
            fprintf(stderr, "vvagent: unknown opcode %u\n", q->op);
            res->code = VOICEVOX_RESULT_INVALID_ACCENT_PHRASE_ERROR;
            break;
    }
}

int main(void) {
    // The agent is a pipe endpoint: line buffering would corrupt the frames.
    setvbuf(stdout, NULL, _IOFBF, 1 << 16);
    fprintf(stderr, "vvagent: ready, core %s\n", voicevox_get_version());
    fflush(stderr);

    for (;;) {
        Req q;
        int r = read_request(&q);
        if (r == 0) break;  // the host went away
        if (r < 0) {
            fprintf(stderr, "vvagent: malformed request\n");
            return 1;
        }
        if (q.op == VV_OP_QUIT) {
            free_request(&q);
            break;
        }
        Res res;
        memset(&res, 0, sizeof res);
        dispatch(&q, &res);
        int w = send_response(&res);
        free_request(&q);
        if (w < 0) break;
    }
    return 0;
}
