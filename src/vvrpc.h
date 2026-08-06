// vvrpc.h - the wire between the host library and the guest agent.
//
// The host exports VOICEVOX CORE's C API; the implementation of every function
// is "send this call to the guest and wait".  The guest is a Linux x86-64
// program running under x86emu with the official libvoicevox_core.so linked in,
// so the work happens where it is meant to.
//
// One frame in, one frame out, over the emulator's stdin and stdout.  The
// emulator writes its own diagnostics to stderr and the core writes its log
// there too, so stdout carries nothing but this.
//
// Both ends are x86-64 and everything is little-endian; no alignment or
// endian conversion is needed or done.
//
//     request   u32 op | u32 nblob | u64 s[4] | nblob * (u32 len, len bytes)
//     response  i32 code | u32 nblob | u64 r[2] | nblob * (u32 len, len bytes)
//
// `code` is a VoicevoxResultCode for the calls that return one, and 0 for the
// calls that do not.  The scalars carry handles, style ids and booleans; the
// blobs carry strings, JSON, UUIDs and WAV data.  A string blob's length never
// includes a terminator - the receiver adds one.
#ifndef VVRPC_H
#define VVRPC_H

#include <stdint.h>

enum VvOp {
    VV_OP_PING = 0,

    // no arguments, answers with a string
    VV_OP_ORT_LIB_VERSIONED_FILENAME = 1,
    VV_OP_ORT_LIB_UNVERSIONED_FILENAME = 2,
    VV_OP_GET_VERSION = 3,

    // the make_default_* constructors, answered as scalars
    VV_OP_DEFAULT_LOAD_ORT_OPTIONS = 4,   // blob0 = default filename
    VV_OP_DEFAULT_INITIALIZE_OPTIONS = 5, // r0 = acceleration_mode, r1 = cpu_num_threads
    VV_OP_DEFAULT_SYNTHESIS_OPTIONS = 6,  // r0 = enable_interrogative_upspeak
    VV_OP_DEFAULT_TTS_OPTIONS = 7,        // r0 = enable_interrogative_upspeak
    VV_OP_DEFAULT_USER_DICT_WORD = 8,     // r0 = word_type, r1 = priority

    // ONNX Runtime
    VV_OP_ORT_GET = 10,        // r0 = handle (0 when not loaded)
    VV_OP_ORT_LOAD_ONCE = 11,  // blob0 = filename -> r0 = handle
    VV_OP_ORT_INIT_ONCE = 12,  // r0 = handle
    VV_OP_ORT_SUPPORTED_DEVICES = 13,  // s0 = ort -> blob0 = json

    // Open JTalk
    VV_OP_OPEN_JTALK_NEW = 20,           // blob0 = dic dir -> r0 = handle
    VV_OP_OPEN_JTALK_USE_USER_DICT = 21, // s0 = ojt, s1 = dict
    VV_OP_OPEN_JTALK_ANALYZE = 22,       // s0 = ojt, blob0 = text -> blob0 = json
    VV_OP_OPEN_JTALK_DELETE = 23,        // s0 = ojt

    // free functions over JSON
    VV_OP_AUDIO_QUERY_FROM_ACCENT_PHRASES = 30,  // blob0 -> blob0
    VV_OP_AUDIO_QUERY_VALIDATE = 31,
    VV_OP_ACCENT_PHRASE_VALIDATE = 32,
    VV_OP_MORA_VALIDATE = 33,
    VV_OP_SCORE_VALIDATE = 34,
    VV_OP_NOTE_VALIDATE = 35,
    VV_OP_FRAME_AUDIO_QUERY_VALIDATE = 36,
    VV_OP_FRAME_PHONEME_VALIDATE = 37,
    VV_OP_ENSURE_COMPATIBLE = 38,  // blob0 = score, blob1 = frame audio query

    // voice model files
    VV_OP_MODEL_OPEN = 40,        // blob0 = path -> r0 = handle
    VV_OP_MODEL_ID = 41,          // s0 = model -> blob0 = 16 bytes
    VV_OP_MODEL_METAS_JSON = 42,  // s0 = model -> blob0 = json
    VV_OP_MODEL_DELETE = 43,      // s0 = model

    // synthesizer
    VV_OP_SYN_NEW = 50,  // s0 = ort, s1 = ojt, s2 = accel, s3 = threads -> r0
    VV_OP_SYN_DELETE = 51,
    VV_OP_SYN_LOAD_VOICE_MODEL = 52,    // s0 = syn, s1 = model
    VV_OP_SYN_UNLOAD_VOICE_MODEL = 53,  // s0 = syn, blob0 = 16-byte id
    VV_OP_SYN_GET_ORT = 54,             // s0 = syn -> r0
    VV_OP_SYN_IS_GPU_MODE = 55,         // s0 = syn -> r0
    VV_OP_SYN_IS_LOADED_VOICE_MODEL = 56,  // s0 = syn, blob0 = id -> r0
    VV_OP_SYN_METAS_JSON = 57,          // s0 = syn -> blob0

    // synthesizer: text in, JSON out
    VV_OP_SYN_AUDIO_QUERY_FROM_KANA = 60,     // s0 = syn, s1 = style, blob0 = kana
    VV_OP_SYN_AUDIO_QUERY = 61,               // s0, s1, blob0 = text
    VV_OP_SYN_ACCENT_PHRASES_FROM_KANA = 62,
    VV_OP_SYN_ACCENT_PHRASES = 63,
    VV_OP_SYN_REPLACE_MORA_DATA = 64,         // blob0 = accent phrases json
    VV_OP_SYN_REPLACE_PHONEME_LENGTH = 65,
    VV_OP_SYN_REPLACE_MORA_PITCH = 66,

    // synthesizer: audio out
    VV_OP_SYN_SYNTHESIS = 70,  // s0 = syn, s1 = style, s2 = upspeak, blob0 = query
    VV_OP_SYN_TTS_FROM_KANA = 71,
    VV_OP_SYN_TTS = 72,
    VV_OP_SYN_FRAME_SYNTHESIS = 73,  // s0 = syn, s1 = style, blob0 = query

    // singing
    VV_OP_SYN_SING_FRAME_AUDIO_QUERY = 80,  // s0, s1 = style, blob0 = score
    VV_OP_SYN_SING_FRAME_F0 = 81,      // blob0 = score, blob1 = frame audio query
    VV_OP_SYN_SING_FRAME_VOLUME = 82,

    // user dictionary
    VV_OP_USER_DICT_NEW = 90,     // -> r0
    VV_OP_USER_DICT_LOAD = 91,    // s0 = dict, blob0 = path
    VV_OP_USER_DICT_ADD_WORD = 92,     // s0, s1 = accent, s2 = type, s3 = priority,
                                       // blob0 = surface, blob1 = pronunciation
                                       // -> blob0 = uuid
    VV_OP_USER_DICT_UPDATE_WORD = 93,  // as above plus blob2 = uuid
    VV_OP_USER_DICT_REMOVE_WORD = 94,  // s0, blob0 = uuid
    VV_OP_USER_DICT_TO_JSON = 95,      // s0 -> blob0
    VV_OP_USER_DICT_IMPORT = 96,       // s0, s1
    VV_OP_USER_DICT_SAVE = 97,         // s0, blob0 = path
    VV_OP_USER_DICT_DELETE = 98,       // s0

    VV_OP_ERROR_MESSAGE = 100,  // s0 = result code -> blob0 = message

    VV_OP_QUIT = 255,
};

#define VV_MAX_SCALARS 4
#define VV_MAX_BLOBS 3
// A WAV for a long utterance is the biggest thing that crosses, and this is a
// ceiling for a corrupt length rather than a budget.
#define VV_MAX_BLOB_BYTES (256u * 1024u * 1024u)

typedef struct VvRequest {
    uint32_t op;
    uint32_t nblob;
    uint64_t s[VV_MAX_SCALARS];
    uint32_t len[VV_MAX_BLOBS];
    const void* blob[VV_MAX_BLOBS];
} VvRequest;

typedef struct VvResponse {
    int32_t code;
    uint32_t nblob;
    uint64_t r[2];
    uint32_t len[VV_MAX_BLOBS];
    void* blob[VV_MAX_BLOBS];
} VvResponse;

#endif  // VVRPC_H
