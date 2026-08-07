// Load a VOICEVOX .vvm through voicevox_onnxruntime, without voicevox_core.
//
//   vvmload <libvoicevox_onnxruntime.so.N> <model.vvm> [which]
//     which: predict_duration (the default) | predict_intonation | decode
//
// What this shows is that the encrypted model format needs nothing from the
// caller except a session option.  A .vvm is a plain ZIP whose manifest.json
// names its members and says what each one is:
//
//   {"vvm_format_version":1, "talk":{
//      "predict_duration":{"type":"vv_bin","filename":"models/pd.bin"}, ...}}
//
// so the caller reads the named member and hands the bytes to
// CreateSessionFromArray with `session.use_vv_bin` set to "1".  The runtime
// does its own decryption, inside itself, and what comes back is an ordinary
// OrtSession.  Nothing here decrypts anything, and nothing here needs to know
// how the payload is encrypted - which is the point: this is the published
// interface being used as published.
//
// Build (Linux):
//   g++ -std=c++17 -Isrc -o vvmload src/vvmload.cpp -ldl
//
// The header is ONNX Runtime's own onnxruntime_c_api.h.
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"

namespace {

const OrtApi* g_api = nullptr;

// The informational calls return a status too.  None of them can fail on a
// session that was already built, but a status is an allocation, so they are
// released rather than ignored - and the compiler is right to insist.
void discard(OrtStatus* st);

bool fail(const char* what, OrtStatus* st) {
    if (!st) return false;
    std::fprintf(stderr, "FAIL  %s: %s\n", what, g_api->GetErrorMessage(st));
    g_api->ReleaseStatus(st);
    return true;
}

void discard(OrtStatus* st) {
    if (st) g_api->ReleaseStatus(st);
}

// ---------------------------------------------------------------------------
// Just enough ZIP to read a member.
//
// Every member of a .vvm is *stored* - the payloads are encrypted and so do not
// compress, and the two json files are tiny - so there is nothing to inflate.
// A deflated member is reported rather than silently mis-read: quietly handing
// compressed bytes to a decryptor would fail somewhere much less obvious.

uint32_t rd32(const std::vector<uint8_t>& b, size_t at) {
    return b[at] | (b[at + 1] << 8) | (b[at + 2] << 16) | ((uint32_t)b[at + 3] << 24);
}
uint16_t rd16(const std::vector<uint8_t>& b, size_t at) {
    return static_cast<uint16_t>(b[at] | (b[at + 1] << 8));
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::vector<uint8_t> out;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n > 0) {
        out.resize(static_cast<size_t>(n));
        if (std::fread(out.data(), 1, out.size(), f) != out.size()) out.clear();
    }
    std::fclose(f);
    return out;
}

// The central directory is at the end, after a signature that has to be
// searched for because the comment field before it is variable length.
bool zip_member(const std::vector<uint8_t>& zip, const std::string& want,
                std::vector<uint8_t>& out) {
    if (zip.size() < 22) return false;
    size_t eocd = 0;
    for (size_t i = zip.size() - 22; i + 1 > 0; i--) {
        if (rd32(zip, i) == 0x06054b50) {
            eocd = i;
            break;
        }
        if (i == 0) break;
    }
    if (!eocd) return false;

    uint16_t count = rd16(zip, eocd + 10);
    size_t at = rd32(zip, eocd + 16);
    for (uint16_t i = 0; i < count && at + 46 <= zip.size(); i++) {
        if (rd32(zip, at) != 0x02014b50) return false;
        uint16_t method = rd16(zip, at + 10);
        uint32_t size = rd32(zip, at + 24);
        uint16_t name_len = rd16(zip, at + 28);
        uint16_t extra_len = rd16(zip, at + 30);
        uint16_t comment_len = rd16(zip, at + 32);
        uint32_t local = rd32(zip, at + 42);
        std::string name(reinterpret_cast<const char*>(&zip[at + 46]), name_len);

        if (name == want) {
            if (method != 0) {
                std::fprintf(stderr, "FAIL  %s is deflated; this reader stores only\n",
                             want.c_str());
                return false;
            }
            // The local header repeats the name and extra fields, and its extra
            // length can differ from the central one - so read it from there.
            if (local + 30 > zip.size() || rd32(zip, local) != 0x04034b50) return false;
            size_t body = local + 30 + rd16(zip, local + 26) + rd16(zip, local + 28);
            if (body + size > zip.size()) return false;
            out.assign(zip.begin() + body, zip.begin() + body + size);
            return true;
        }
        at += 46 + name_len + extra_len + comment_len;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Just enough JSON to find one filename.
//
// The manifest is generated, small and regular, so this looks for
// `"<which>"` and then the `"filename"` that follows it.  A general parser
// would be a lot of code to answer one question.

std::string manifest_filename(const std::vector<uint8_t>& manifest,
                              const std::string& which) {
    std::string s(reinterpret_cast<const char*>(manifest.data()), manifest.size());
    size_t at = s.find("\"" + which + "\"");
    if (at == std::string::npos) return {};
    size_t key = s.find("\"filename\"", at);
    if (key == std::string::npos) return {};
    size_t open = s.find('"', s.find(':', key) + 1);
    size_t close = s.find('"', open + 1);
    if (open == std::string::npos || close == std::string::npos) return {};
    return s.substr(open + 1, close - open - 1);
}

const char* type_name(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
        default: return "?";
    }
}

void describe(OrtSession* session, OrtAllocator* alloc) {
    size_t n_in = 0, n_out = 0;
    discard(g_api->SessionGetInputCount(session, &n_in));
    discard(g_api->SessionGetOutputCount(session, &n_out));
    std::printf("      %zu inputs, %zu outputs\n", n_in, n_out);

    for (int side = 0; side < 2; side++) {
        size_t n = side ? n_out : n_in;
        for (size_t i = 0; i < n; i++) {
            char* name = nullptr;
            OrtTypeInfo* info = nullptr;
            if (side) {
                discard(g_api->SessionGetOutputName(session, i, alloc, &name));
                discard(g_api->SessionGetOutputTypeInfo(session, i, &info));
            } else {
                discard(g_api->SessionGetInputName(session, i, alloc, &name));
                discard(g_api->SessionGetInputTypeInfo(session, i, &info));
            }
            const OrtTensorTypeAndShapeInfo* shape = nullptr;
            discard(g_api->CastTypeInfoToTensorInfo(info, &shape));
            ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            size_t dims = 0;
            std::string text;
            if (shape) {
                discard(g_api->GetTensorElementType(shape, &type));
                discard(g_api->GetDimensionsCount(shape, &dims));
                std::vector<int64_t> d(dims);
                discard(g_api->GetDimensions(shape, d.data(), dims));
                for (size_t k = 0; k < dims; k++)
                    text += (k ? "," : "") + (d[k] < 0 ? std::string("?")
                                                       : std::to_string(d[k]));
            }
            std::printf("      %-7s %-18s %-6s [%s]\n", side ? "output" : "input", name,
                        type_name(type), text.c_str());
            g_api->ReleaseTypeInfo(info);
            alloc->Free(alloc, name);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: vvmload <libvoicevox_onnxruntime.so.N> <model.vvm> "
                     "[predict_duration|predict_intonation|decode]\n");
        return 2;
    }
    const std::string runtime = argv[1];
    const std::string vvm = argv[2];
    const std::string which = argc > 3 ? argv[3] : "predict_duration";

    // ---- the runtime -------------------------------------------------------
    // dlopen rather than linking, because there is more than one build of it -
    // CPU and CUDA - and which one is in use is a run-time choice.
    std::printf("step  dlopen %s\n", runtime.c_str());
    void* lib = dlopen(runtime.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        std::fprintf(stderr, "FAIL  dlopen: %s\n", dlerror());
        return 1;
    }
    auto get_base = reinterpret_cast<const OrtApiBase* (*)()>(dlsym(lib, "OrtGetApiBase"));
    if (!get_base) {
        std::fprintf(stderr, "FAIL  no OrtGetApiBase - is this voicevox_onnxruntime?\n");
        return 1;
    }
    const OrtApiBase* base = get_base();
    g_api = base->GetApi(ORT_API_VERSION);
    if (!g_api) {
        std::fprintf(stderr, "FAIL  this build does not speak API version %d\n",
                     ORT_API_VERSION);
        return 1;
    }
    std::printf("ok    %s\n", base->GetVersionString());

    // ---- the model out of the container ------------------------------------
    std::printf("step  open %s\n", vvm.c_str());
    std::vector<uint8_t> zip = read_file(vvm);
    if (zip.empty()) {
        std::fprintf(stderr, "FAIL  cannot read %s\n", vvm.c_str());
        return 1;
    }
    std::vector<uint8_t> manifest;
    if (!zip_member(zip, "manifest.json", manifest)) {
        std::fprintf(stderr, "FAIL  no manifest.json - is this a .vvm?\n");
        return 1;
    }
    std::string member = manifest_filename(manifest, which);
    if (member.empty()) {
        std::fprintf(stderr, "FAIL  the manifest has no %s\n", which.c_str());
        return 1;
    }
    std::vector<uint8_t> model;
    if (!zip_member(zip, member, model)) {
        std::fprintf(stderr, "FAIL  cannot read %s from the container\n", member.c_str());
        return 1;
    }
    std::printf("ok    %s -> %s, %zu bytes (still encrypted)\n", which.c_str(),
                member.c_str(), model.size());

    // ---- the session -------------------------------------------------------
    OrtEnv* env = nullptr;
    if (fail("CreateEnv", g_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "vvmload", &env)))
        return 1;

    OrtSessionOptions* so = nullptr;
    if (fail("CreateSessionOptions", g_api->CreateSessionOptions(&so))) return 1;

    // This is the whole of it.  Without the entry the runtime sees a payload it
    // does not recognise as ONNX and refuses it; with it, it decrypts the
    // payload itself before parsing.  The caller never holds plaintext.
    std::printf("step  session.use_vv_bin = 1\n");
    if (fail("AddSessionConfigEntry",
             g_api->AddSessionConfigEntry(so, "session.use_vv_bin", "1")))
        return 1;

    std::printf("step  CreateSessionFromArray  <- the runtime decrypts, inside itself\n");
    OrtSession* session = nullptr;
    if (fail("CreateSessionFromArray",
             g_api->CreateSessionFromArray(env, model.data(), model.size(), so, &session)))
        return 1;
    std::printf("ok    session built\n");

    OrtAllocator* alloc = nullptr;
    discard(g_api->GetAllocatorWithDefaultOptions(&alloc));
    describe(session, alloc);

    g_api->ReleaseSession(session);
    g_api->ReleaseSessionOptions(so);
    g_api->ReleaseEnv(env);
    std::printf("VVMLOAD OK\n");
    return 0;
}
