// vvhost.c - VOICEVOX CORE's C API, implemented by an emulator.
//
// Every function here has the signature the official `voicevox_core.h` gives
// it, and the same meaning.  What is different is underneath: instead of a
// native libvoicevox_core, the call is packed into a frame, handed to a Linux
// x86-64 guest running under x86emu, and answered by the official
// libvoicevox_core.so making the same call for real.
//
// So a program written against VOICEVOX CORE links this instead and runs
// unchanged - on a host where no build of CORE exists, which is the point.
//
// Handles (OpenJtalkRc*, VoicevoxSynthesizer*, ...) are guest pointers passed
// through untouched.  They are opaque on both sides, both sides are 64-bit, and
// nothing here ever dereferences one.
//
// Buffers the API says the caller must free - JSON and WAV - are copied into
// the host heap and freed by the host's own voicevox_json_free /
// voicevox_wav_free.  The guest frees its own copy as soon as it has been sent,
// so no allocation outlives a call on either side.
//
// Paths are the one place the two worlds do not line up, and `to_guest_path`
// below says what is done about it.

#if defined(_WIN32)
// The published header marks every function __declspec(dllimport), which is
// right for the callers and wrong for the definitions.  `dllimport` inside
// __declspec is an ordinary identifier, so one macro turns the whole header
// into the export side without a second copy of it to keep in step.
#define dllimport dllexport
#endif

#include "voicevox_core.h"

#if defined(_WIN32)
#undef dllimport
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vvrpc.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// where the pieces are

// A clone's layout, which is also the default:
//
//     <root>/x86emu.exe            the emulator            VOICEVOX_EMU_EMULATOR
//     <root>/sysroot/              the guest's filesystem  VOICEVOX_EMU_SYSROOT
//     <root>/sysroot/opt/vv/vvagent  the guest agent       VOICEVOX_EMU_AGENT
//
// <root> is VOICEVOX_EMU_ROOT if set, else the directory this library was
// loaded from, else the working directory.
static char g_root[1024];
static char g_emulator[1200];
static char g_sysroot[1200];
static char g_agent[512] = "/opt/vv/vvagent";
static char g_last_error[512];

static void set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof g_last_error, fmt, ap);
    va_end(ap);
    fprintf(stderr, "voicevox_emu: %s\n", g_last_error);
    fflush(stderr);
}

static int env_copy(const char* name, char* out, size_t n) {
    const char* v = getenv(name);
    if (!v || !*v) return 0;
    snprintf(out, n, "%s", v);
    return 1;
}

#if defined(_WIN32)
static HMODULE self_module(void) {
    HMODULE h = NULL;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)(void*)&self_module, &h);
    return h;
}
#endif

static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void discover_layout(void) {
    if (g_emulator[0]) return;  // already done

    if (!env_copy("VOICEVOX_EMU_ROOT", g_root, sizeof g_root)) {
#if defined(_WIN32)
        char mod[1024] = {0};
        if (GetModuleFileNameA(self_module(), mod, sizeof mod - 1)) {
            char* slash = strrchr(mod, '\\');
            if (slash) *slash = 0;
            snprintf(g_root, sizeof g_root, "%s", mod);
        } else {
            snprintf(g_root, sizeof g_root, ".");
        }
#else
        snprintf(g_root, sizeof g_root, ".");
#endif
    }

    if (!env_copy("VOICEVOX_EMU_EMULATOR", g_emulator, sizeof g_emulator)) {
#if defined(_WIN32)
        snprintf(g_emulator, sizeof g_emulator, "%s\\x86emu.exe", g_root);
        if (!file_exists(g_emulator))
            snprintf(g_emulator, sizeof g_emulator, "%s\\x86_emu_cpp\\x86emu.exe", g_root);
#else
        snprintf(g_emulator, sizeof g_emulator, "%s/x86emu", g_root);
        if (!file_exists(g_emulator))
            snprintf(g_emulator, sizeof g_emulator, "%s/x86_emu_cpp/x86emu", g_root);
#endif
    }
    if (!env_copy("VOICEVOX_EMU_SYSROOT", g_sysroot, sizeof g_sysroot)) {
#if defined(_WIN32)
        snprintf(g_sysroot, sizeof g_sysroot, "%s\\sysroot", g_root);
#else
        snprintf(g_sysroot, sizeof g_sysroot, "%s/sysroot", g_root);
#endif
    }
    env_copy("VOICEVOX_EMU_AGENT", g_agent, sizeof g_agent);
}

// A host path the caller handed us has to become a path the guest can open,
// and the guest's filesystem is the sysroot directory and nothing else.
//
// - A path already inside the sysroot becomes its guest-absolute form:
//   `C:\vv\sysroot\opt\vv\0.vvm` -> `/opt/vv/0.vvm`.
// - A path that already looks guest-absolute (`/opt/vv/0.vvm`) is left alone,
//   so a caller who knows where it is putting things can say so directly.
// - Anything else is passed through unchanged and will fail in the guest with
//   the error the API defines for a missing file, which is the honest answer:
//   the guest cannot see outside its filesystem, and silently substituting
//   something would be worse than saying so.
static const char* to_guest_path(const char* path, char* buf, size_t n) {
    if (!path) return path;
    discover_layout();
    size_t rootlen = strlen(g_sysroot);
    if (rootlen && strlen(path) > rootlen) {
        int match = 1;
        for (size_t i = 0; i < rootlen; i++) {
            char a = path[i], b = g_sysroot[i];
#if defined(_WIN32)
            if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
            if (a == '\\') a = '/';
            if (b == '\\') b = '/';
#endif
            if (a != b) {
                match = 0;
                break;
            }
        }
        if (match && (path[rootlen] == '/' || path[rootlen] == '\\')) {
            size_t j = 0;
            buf[j++] = '/';
            for (size_t i = rootlen + 1; path[i] && j + 1 < n; i++)
                buf[j++] = path[i] == '\\' ? '/' : path[i];
            buf[j] = 0;
            return buf;
        }
    }
    return path;
}

// ---------------------------------------------------------------------------
// the child process and the one call at a time that goes through it

typedef struct {
    int up;
    int broken;
#if defined(_WIN32)
    HANDLE proc;
    HANDLE to_child;    // we write requests here
    HANDLE from_child;  // we read responses here
    CRITICAL_SECTION lock;
    int lock_ready;
#else
    pid_t pid;
    int to_child;
    int from_child;
    pthread_mutex_t lock;
#endif
} Agent;

// Zero is the right start for every field on both platforms; the POSIX fds are
// set before they are used and the critical section is created on first lock.
static Agent g_agent_state;

static void agent_lock(void) {
#if defined(_WIN32)
    static LONG once = 0;
    if (InterlockedCompareExchange(&once, 1, 0) == 0) {
        InitializeCriticalSection(&g_agent_state.lock);
        g_agent_state.lock_ready = 1;
    }
    while (!g_agent_state.lock_ready) Sleep(0);
    EnterCriticalSection(&g_agent_state.lock);
#else
    pthread_mutex_lock(&g_agent_state.lock);
#endif
}

static void agent_unlock(void) {
#if defined(_WIN32)
    LeaveCriticalSection(&g_agent_state.lock);
#else
    pthread_mutex_unlock(&g_agent_state.lock);
#endif
}

static int agent_start(void) {
    if (g_agent_state.up) return 1;
    if (g_agent_state.broken) return 0;
    discover_layout();

    if (!file_exists(g_emulator)) {
        set_error("no emulator at %s (set VOICEVOX_EMU_EMULATOR or VOICEVOX_EMU_ROOT)",
                  g_emulator);
        g_agent_state.broken = 1;
        return 0;
    }

#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof sa;
    sa.lpSecurityDescriptor = NULL;
    sa.bInheritHandle = TRUE;

    HANDLE in_r = NULL, in_w = NULL, out_r = NULL, out_w = NULL;
    if (!CreatePipe(&in_r, &in_w, &sa, 0) || !CreatePipe(&out_r, &out_w, &sa, 0)) {
        set_error("CreatePipe failed (%lu)", (unsigned long)GetLastError());
        g_agent_state.broken = 1;
        return 0;
    }
    // Only the child's ends may be inherited, or a read here never sees EOF.
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);

    char cmd[4096];
    snprintf(cmd, sizeof cmd, "\"%s\" --sysroot \"%s\" \"%s%s\" ", g_emulator, g_sysroot,
             g_sysroot, g_agent);

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof si);
    memset(&pi, 0, sizeof pi);
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r;
    si.hStdOutput = out_w;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    BOOL ok = CreateProcessA(NULL, cmd, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, g_root, &si, &pi);
    CloseHandle(in_r);
    CloseHandle(out_w);
    if (!ok) {
        CloseHandle(in_w);
        CloseHandle(out_r);
        set_error("cannot start %s (%lu)", g_emulator, (unsigned long)GetLastError());
        g_agent_state.broken = 1;
        return 0;
    }
    CloseHandle(pi.hThread);
    g_agent_state.proc = pi.hProcess;
    g_agent_state.to_child = in_w;
    g_agent_state.from_child = out_r;
#else
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) < 0 || pipe(out_pipe) < 0) {
        set_error("pipe: %s", strerror(errno));
        g_agent_state.broken = 1;
        return 0;
    }
    pid_t pid = fork();
    if (pid < 0) {
        set_error("fork: %s", strerror(errno));
        g_agent_state.broken = 1;
        return 0;
    }
    if (pid == 0) {
        dup2(in_pipe[0], 0);
        dup2(out_pipe[1], 1);
        close(in_pipe[0]);
        close(in_pipe[1]);
        close(out_pipe[0]);
        close(out_pipe[1]);
        char guest_prog[2048];
        snprintf(guest_prog, sizeof guest_prog, "%s%s", g_sysroot, g_agent);
        execl(g_emulator, g_emulator, "--sysroot", g_sysroot, guest_prog, (char*)NULL);
        _exit(127);
    }
    close(in_pipe[0]);
    close(out_pipe[1]);
    g_agent_state.pid = pid;
    g_agent_state.to_child = in_pipe[1];
    g_agent_state.from_child = out_pipe[0];
#endif
    g_agent_state.up = 1;
    return 1;
}

static int io_write(const void* p, size_t n) {
    const unsigned char* b = p;
#if defined(_WIN32)
    while (n) {
        DWORD put = 0;
        if (!WriteFile(g_agent_state.to_child, b, (DWORD)n, &put, NULL) || put == 0) return -1;
        b += put;
        n -= put;
    }
#else
    while (n) {
        ssize_t put = write(g_agent_state.to_child, b, n);
        if (put <= 0) return -1;
        b += put;
        n -= (size_t)put;
    }
#endif
    return 0;
}

static int io_read(void* p, size_t n) {
    unsigned char* b = p;
#if defined(_WIN32)
    while (n) {
        DWORD got = 0;
        if (!ReadFile(g_agent_state.from_child, b, (DWORD)n, &got, NULL) || got == 0) return -1;
        b += got;
        n -= got;
    }
#else
    while (n) {
        ssize_t got = read(g_agent_state.from_child, b, n);
        if (got <= 0) return -1;
        b += got;
        n -= (size_t)got;
    }
#endif
    return 0;
}

// One request out, one response in.  `out` owns any blobs on success and the
// caller frees them with free().
static int rpc(const VvRequest* req, VvResponse* out) {
    memset(out, 0, sizeof *out);
    agent_lock();
    if (!agent_start()) {
        agent_unlock();
        return 0;
    }

    uint32_t head[2] = {req->op, req->nblob};
    int bad = io_write(head, sizeof head) < 0 || io_write(req->s, sizeof req->s) < 0;
    for (uint32_t i = 0; !bad && i < req->nblob; i++) {
        bad = io_write(&req->len[i], 4) < 0;
        if (!bad && req->len[i]) bad = io_write(req->blob[i], req->len[i]) < 0;
    }
    if (bad) {
        set_error("the guest agent stopped accepting requests");
        g_agent_state.up = 0;
        g_agent_state.broken = 1;
        agent_unlock();
        return 0;
    }

    uint32_t rhead[2];
    if (io_read(rhead, sizeof rhead) < 0 || io_read(out->r, sizeof out->r) < 0) {
        set_error("the guest agent gave no answer (it may have died - see stderr)");
        g_agent_state.up = 0;
        g_agent_state.broken = 1;
        agent_unlock();
        return 0;
    }
    memcpy(&out->code, &rhead[0], 4);
    out->nblob = rhead[1];
    if (out->nblob > VV_MAX_BLOBS) {
        set_error("the guest agent sent a malformed frame");
        g_agent_state.up = 0;
        g_agent_state.broken = 1;
        agent_unlock();
        return 0;
    }
    for (uint32_t i = 0; i < out->nblob; i++) {
        if (io_read(&out->len[i], 4) < 0 || out->len[i] > VV_MAX_BLOB_BYTES) {
            for (uint32_t j = 0; j < i; j++) free(out->blob[j]);
            memset(out, 0, sizeof *out);
            set_error("the guest agent sent a malformed blob");
            g_agent_state.up = 0;
            g_agent_state.broken = 1;
            agent_unlock();
            return 0;
        }
        // One spare byte so a JSON or message blob can be used as a C string.
        out->blob[i] = malloc((size_t)out->len[i] + 1);
        if (!out->blob[i]) {
            for (uint32_t j = 0; j < i; j++) free(out->blob[j]);
            memset(out, 0, sizeof *out);
            agent_unlock();
            return 0;
        }
        if (out->len[i] && io_read(out->blob[i], out->len[i]) < 0) {
            for (uint32_t j = 0; j <= i; j++) free(out->blob[j]);
            memset(out, 0, sizeof *out);
            set_error("the guest agent's answer was cut short");
            g_agent_state.up = 0;
            g_agent_state.broken = 1;
            agent_unlock();
            return 0;
        }
        ((char*)out->blob[i])[out->len[i]] = 0;
    }
    agent_unlock();
    return 1;
}

static void free_response(VvResponse* r) {
    for (uint32_t i = 0; i < r->nblob; i++) free(r->blob[i]);
    r->nblob = 0;
}

// The code every entry point returns when the emulator could not be reached.
// Nothing was run, so this is the "the runtime is not there" answer.
#define VV_UNREACHABLE VOICEVOX_RESULT_INIT_INFERENCE_RUNTIME_ERROR

// ---- small helpers over rpc() ---------------------------------------------

static int call_simple(uint32_t op, const uint64_t* s, int ns, VvResponse* res) {
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = op;
    for (int i = 0; i < ns && i < VV_MAX_SCALARS; i++) req.s[i] = s[i];
    return rpc(&req, res);
}

// Calls that answer with a string the API says lives forever: cached, once.
static const char* cached_string(uint32_t op, const char** slot) {
    if (*slot) return *slot;
    VvResponse res;
    uint64_t s[1] = {0};
    if (!call_simple(op, s, 0, &res)) return "";
    char* copy = malloc((size_t)res.len[0] + 1);
    if (copy) {
        memcpy(copy, res.blob[0], res.len[0]);
        copy[res.len[0]] = 0;
        *slot = copy;
    }
    free_response(&res);
    return *slot ? *slot : "";
}

// (synthesizer, string, style) -> JSON, which seven functions share.
static VoicevoxResultCode json_from_text(uint32_t op, const void* handle, const char* text,
                                         VoicevoxStyleId style, char** out_json) {
    if (out_json) *out_json = NULL;
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = op;
    req.s[0] = (uint64_t)(uintptr_t)handle;
    req.s[1] = style;
    req.nblob = 1;
    req.blob[0] = text ? text : "";
    req.len[0] = (uint32_t)strlen(text ? text : "");
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    if (res.code == VOICEVOX_RESULT_OK && out_json && res.nblob) {
        *out_json = malloc((size_t)res.len[0] + 1);
        if (*out_json) {
            memcpy(*out_json, res.blob[0], res.len[0]);
            (*out_json)[res.len[0]] = 0;
        }
    }
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

// (one JSON string in) -> a result code, which the validators share.
static VoicevoxResultCode validate_json(uint32_t op, const char* json) {
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = op;
    req.nblob = 1;
    req.blob[0] = json ? json : "";
    req.len[0] = (uint32_t)strlen(json ? json : "");
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

static VoicevoxResultCode wav_call(uint32_t op, const void* syn, const char* text,
                                   VoicevoxStyleId style, int upspeak,
                                   uintptr_t* out_len, uint8_t** out_wav) {
    if (out_len) *out_len = 0;
    if (out_wav) *out_wav = NULL;
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = op;
    req.s[0] = (uint64_t)(uintptr_t)syn;
    req.s[1] = style;
    req.s[2] = (uint64_t)upspeak;
    req.nblob = 1;
    req.blob[0] = text ? text : "";
    req.len[0] = (uint32_t)strlen(text ? text : "");
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    if (res.code == VOICEVOX_RESULT_OK && res.nblob && out_wav && out_len) {
        *out_wav = malloc(res.len[0] ? res.len[0] : 1);
        if (*out_wav) {
            memcpy(*out_wav, res.blob[0], res.len[0]);
            *out_len = res.len[0];
        }
    }
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

// ---------------------------------------------------------------------------
// the API

const char* voicevox_get_onnxruntime_lib_versioned_filename(void) {
    static const char* s;
    return cached_string(VV_OP_ORT_LIB_VERSIONED_FILENAME, &s);
}

const char* voicevox_get_onnxruntime_lib_unversioned_filename(void) {
    static const char* s;
    return cached_string(VV_OP_ORT_LIB_UNVERSIONED_FILENAME, &s);
}

const char* voicevox_get_version(void) {
    static const char* s;
    return cached_string(VV_OP_GET_VERSION, &s);
}

struct VoicevoxLoadOnnxruntimeOptions voicevox_make_default_load_onnxruntime_options(void) {
    static const char* s;
    struct VoicevoxLoadOnnxruntimeOptions o;
    o.filename = cached_string(VV_OP_DEFAULT_LOAD_ORT_OPTIONS, &s);
    return o;
}

struct VoicevoxInitializeOptions voicevox_make_default_initialize_options(void) {
    static int have;
    static struct VoicevoxInitializeOptions cached;
    if (!have) {
        VvResponse res;
        if (call_simple(VV_OP_DEFAULT_INITIALIZE_OPTIONS, NULL, 0, &res)) {
            cached.acceleration_mode = (VoicevoxAccelerationMode)(int32_t)(uint32_t)res.r[0];
            cached.cpu_num_threads = (uint16_t)res.r[1];
            have = 1;
            free_response(&res);
        } else {
            cached.acceleration_mode = VOICEVOX_ACCELERATION_MODE_AUTO;
            cached.cpu_num_threads = 0;
        }
    }
    return cached;
}

struct VoicevoxSynthesisOptions voicevox_make_default_synthesis_options(void) {
    static int have;
    static struct VoicevoxSynthesisOptions cached;
    if (!have) {
        VvResponse res;
        if (call_simple(VV_OP_DEFAULT_SYNTHESIS_OPTIONS, NULL, 0, &res)) {
            cached.enable_interrogative_upspeak = res.r[0] != 0;
            have = 1;
            free_response(&res);
        } else {
            cached.enable_interrogative_upspeak = false;
        }
    }
    return cached;
}

struct VoicevoxTtsOptions voicevox_make_default_tts_options(void) {
    static int have;
    static struct VoicevoxTtsOptions cached;
    if (!have) {
        VvResponse res;
        if (call_simple(VV_OP_DEFAULT_TTS_OPTIONS, NULL, 0, &res)) {
            cached.enable_interrogative_upspeak = res.r[0] != 0;
            have = 1;
            free_response(&res);
        } else {
            cached.enable_interrogative_upspeak = true;
        }
    }
    return cached;
}

const struct VoicevoxOnnxruntime* voicevox_onnxruntime_get(void) {
    VvResponse res;
    if (!call_simple(VV_OP_ORT_GET, NULL, 0, &res)) return NULL;
    const struct VoicevoxOnnxruntime* p =
        (const struct VoicevoxOnnxruntime*)(uintptr_t)res.r[0];
    free_response(&res);
    return p;
}

VoicevoxResultCode voicevox_onnxruntime_load_once(
    struct VoicevoxLoadOnnxruntimeOptions options,
    const struct VoicevoxOnnxruntime** out_onnxruntime) {
    if (out_onnxruntime) *out_onnxruntime = NULL;
    char buf[2048];
    const char* fname = options.filename ? to_guest_path(options.filename, buf, sizeof buf) : NULL;
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = VV_OP_ORT_LOAD_ONCE;
    if (fname) {
        req.nblob = 1;
        req.blob[0] = fname;
        req.len[0] = (uint32_t)strlen(fname);
    }
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    if (out_onnxruntime)
        *out_onnxruntime = (const struct VoicevoxOnnxruntime*)(uintptr_t)res.r[0];
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

#if defined(VOICEVOX_LINK_ONNXRUNTIME)
VoicevoxResultCode voicevox_onnxruntime_init_once(
    const struct VoicevoxOnnxruntime** out_onnxruntime) {
    if (out_onnxruntime) *out_onnxruntime = NULL;
    VvResponse res;
    if (!call_simple(VV_OP_ORT_INIT_ONCE, NULL, 0, &res)) return VV_UNREACHABLE;
    if (out_onnxruntime)
        *out_onnxruntime = (const struct VoicevoxOnnxruntime*)(uintptr_t)res.r[0];
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}
#endif  // VOICEVOX_LINK_ONNXRUNTIME

VoicevoxResultCode voicevox_onnxruntime_create_supported_devices_json(
    const struct VoicevoxOnnxruntime* onnxruntime, char** output_supported_devices_json) {
    if (output_supported_devices_json) *output_supported_devices_json = NULL;
    uint64_t s[1] = {(uint64_t)(uintptr_t)onnxruntime};
    VvResponse res;
    if (!call_simple(VV_OP_ORT_SUPPORTED_DEVICES, s, 1, &res)) return VV_UNREACHABLE;
    if (res.code == VOICEVOX_RESULT_OK && output_supported_devices_json && res.nblob) {
        *output_supported_devices_json = malloc((size_t)res.len[0] + 1);
        if (*output_supported_devices_json) {
            memcpy(*output_supported_devices_json, res.blob[0], res.len[0]);
            (*output_supported_devices_json)[res.len[0]] = 0;
        }
    }
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_open_jtalk_rc_new(const char* open_jtalk_dic_dir,
                                              struct OpenJtalkRc** out_open_jtalk) {
    if (out_open_jtalk) *out_open_jtalk = NULL;
    char buf[2048];
    const char* dir = to_guest_path(open_jtalk_dic_dir, buf, sizeof buf);
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = VV_OP_OPEN_JTALK_NEW;
    req.nblob = 1;
    req.blob[0] = dir ? dir : "";
    req.len[0] = (uint32_t)strlen(dir ? dir : "");
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    if (out_open_jtalk) *out_open_jtalk = (struct OpenJtalkRc*)(uintptr_t)res.r[0];
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_open_jtalk_rc_use_user_dict(const struct OpenJtalkRc* open_jtalk,
                                                        const struct VoicevoxUserDict* user_dict) {
    uint64_t s[2] = {(uint64_t)(uintptr_t)open_jtalk, (uint64_t)(uintptr_t)user_dict};
    VvResponse res;
    if (!call_simple(VV_OP_OPEN_JTALK_USE_USER_DICT, s, 2, &res)) return VV_UNREACHABLE;
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_open_jtalk_rc_analyze(const struct OpenJtalkRc* open_jtalk,
                                                  const char* text,
                                                  char** output_accent_phrases_json) {
    return json_from_text(VV_OP_OPEN_JTALK_ANALYZE, open_jtalk, text, 0,
                          output_accent_phrases_json);
}

void voicevox_open_jtalk_rc_delete(struct OpenJtalkRc* open_jtalk) {
    uint64_t s[1] = {(uint64_t)(uintptr_t)open_jtalk};
    VvResponse res;
    if (call_simple(VV_OP_OPEN_JTALK_DELETE, s, 1, &res)) free_response(&res);
}

VoicevoxResultCode voicevox_audio_query_create_from_accent_phrases(
    const char* accent_phrases_json, char** output_audio_query_json) {
    return json_from_text(VV_OP_AUDIO_QUERY_FROM_ACCENT_PHRASES, NULL, accent_phrases_json, 0,
                          output_audio_query_json);
}

VoicevoxResultCode voicevox_audio_query_validate(const char* audio_query_json) {
    return validate_json(VV_OP_AUDIO_QUERY_VALIDATE, audio_query_json);
}
VoicevoxResultCode voicevox_accent_phrase_validate(const char* accent_phrase_json) {
    return validate_json(VV_OP_ACCENT_PHRASE_VALIDATE, accent_phrase_json);
}
VoicevoxResultCode voicevox_mora_validate(const char* mora_json) {
    return validate_json(VV_OP_MORA_VALIDATE, mora_json);
}
VoicevoxResultCode voicevox_score_validate(const char* score_json) {
    return validate_json(VV_OP_SCORE_VALIDATE, score_json);
}
VoicevoxResultCode voicevox_note_validate(const char* note_json) {
    return validate_json(VV_OP_NOTE_VALIDATE, note_json);
}
VoicevoxResultCode voicevox_frame_audio_query_validate(const char* frame_audio_query_json) {
    return validate_json(VV_OP_FRAME_AUDIO_QUERY_VALIDATE, frame_audio_query_json);
}
VoicevoxResultCode voicevox_frame_phoneme_validate(const char* frame_phoneme_json) {
    return validate_json(VV_OP_FRAME_PHONEME_VALIDATE, frame_phoneme_json);
}

VoicevoxResultCode voicevox_ensure_compatible(const char* score_json,
                                              const char* frame_audio_query_json) {
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = VV_OP_ENSURE_COMPATIBLE;
    req.nblob = 2;
    req.blob[0] = score_json ? score_json : "";
    req.len[0] = (uint32_t)strlen(score_json ? score_json : "");
    req.blob[1] = frame_audio_query_json ? frame_audio_query_json : "";
    req.len[1] = (uint32_t)strlen(frame_audio_query_json ? frame_audio_query_json : "");
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_voice_model_file_open(const char* path,
                                                  struct VoicevoxVoiceModelFile** out_model) {
    if (out_model) *out_model = NULL;
    char buf[2048];
    const char* p = to_guest_path(path, buf, sizeof buf);
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = VV_OP_MODEL_OPEN;
    req.nblob = 1;
    req.blob[0] = p ? p : "";
    req.len[0] = (uint32_t)strlen(p ? p : "");
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    if (out_model) *out_model = (struct VoicevoxVoiceModelFile*)(uintptr_t)res.r[0];
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

void voicevox_voice_model_file_id(const struct VoicevoxVoiceModelFile* model,
                                  uint8_t (*output_voice_model_id)[16]) {
    if (output_voice_model_id) memset(*output_voice_model_id, 0, 16);
    uint64_t s[1] = {(uint64_t)(uintptr_t)model};
    VvResponse res;
    if (!call_simple(VV_OP_MODEL_ID, s, 1, &res)) return;
    if (output_voice_model_id && res.nblob && res.len[0] == 16)
        memcpy(*output_voice_model_id, res.blob[0], 16);
    free_response(&res);
}

char* voicevox_voice_model_file_create_metas_json(const struct VoicevoxVoiceModelFile* model) {
    uint64_t s[1] = {(uint64_t)(uintptr_t)model};
    VvResponse res;
    if (!call_simple(VV_OP_MODEL_METAS_JSON, s, 1, &res)) return NULL;
    char* json = NULL;
    if (res.nblob) {
        json = malloc((size_t)res.len[0] + 1);
        if (json) {
            memcpy(json, res.blob[0], res.len[0]);
            json[res.len[0]] = 0;
        }
    }
    free_response(&res);
    return json;
}

void voicevox_voice_model_file_delete(struct VoicevoxVoiceModelFile* model) {
    uint64_t s[1] = {(uint64_t)(uintptr_t)model};
    VvResponse res;
    if (call_simple(VV_OP_MODEL_DELETE, s, 1, &res)) free_response(&res);
}

VoicevoxResultCode voicevox_synthesizer_new(const struct VoicevoxOnnxruntime* onnxruntime,
                                            const struct OpenJtalkRc* open_jtalk,
                                            struct VoicevoxInitializeOptions options,
                                            struct VoicevoxSynthesizer** out_synthesizer) {
    if (out_synthesizer) *out_synthesizer = NULL;
    uint64_t s[4] = {(uint64_t)(uintptr_t)onnxruntime, (uint64_t)(uintptr_t)open_jtalk,
                     (uint64_t)(uint32_t)options.acceleration_mode, options.cpu_num_threads};
    VvResponse res;
    if (!call_simple(VV_OP_SYN_NEW, s, 4, &res)) return VV_UNREACHABLE;
    if (out_synthesizer) *out_synthesizer = (struct VoicevoxSynthesizer*)(uintptr_t)res.r[0];
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

void voicevox_synthesizer_delete(struct VoicevoxSynthesizer* synthesizer) {
    uint64_t s[1] = {(uint64_t)(uintptr_t)synthesizer};
    VvResponse res;
    if (call_simple(VV_OP_SYN_DELETE, s, 1, &res)) free_response(&res);
}

VoicevoxResultCode voicevox_synthesizer_load_voice_model(
    const struct VoicevoxSynthesizer* synthesizer, const struct VoicevoxVoiceModelFile* model) {
    uint64_t s[2] = {(uint64_t)(uintptr_t)synthesizer, (uint64_t)(uintptr_t)model};
    VvResponse res;
    if (!call_simple(VV_OP_SYN_LOAD_VOICE_MODEL, s, 2, &res)) return VV_UNREACHABLE;
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_synthesizer_unload_voice_model(
    const struct VoicevoxSynthesizer* synthesizer, VoicevoxVoiceModelId model_id) {
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = VV_OP_SYN_UNLOAD_VOICE_MODEL;
    req.s[0] = (uint64_t)(uintptr_t)synthesizer;
    req.nblob = 1;
    req.blob[0] = model_id;
    req.len[0] = 16;
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

const struct VoicevoxOnnxruntime* voicevox_synthesizer_get_onnxruntime(
    const struct VoicevoxSynthesizer* synthesizer) {
    uint64_t s[1] = {(uint64_t)(uintptr_t)synthesizer};
    VvResponse res;
    if (!call_simple(VV_OP_SYN_GET_ORT, s, 1, &res)) return NULL;
    const struct VoicevoxOnnxruntime* p =
        (const struct VoicevoxOnnxruntime*)(uintptr_t)res.r[0];
    free_response(&res);
    return p;
}

bool voicevox_synthesizer_is_gpu_mode(const struct VoicevoxSynthesizer* synthesizer) {
    uint64_t s[1] = {(uint64_t)(uintptr_t)synthesizer};
    VvResponse res;
    if (!call_simple(VV_OP_SYN_IS_GPU_MODE, s, 1, &res)) return false;
    bool v = res.r[0] != 0;
    free_response(&res);
    return v;
}

bool voicevox_synthesizer_is_loaded_voice_model(const struct VoicevoxSynthesizer* synthesizer,
                                                VoicevoxVoiceModelId model_id) {
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = VV_OP_SYN_IS_LOADED_VOICE_MODEL;
    req.s[0] = (uint64_t)(uintptr_t)synthesizer;
    req.nblob = 1;
    req.blob[0] = model_id;
    req.len[0] = 16;
    VvResponse res;
    if (!rpc(&req, &res)) return false;
    bool v = res.r[0] != 0;
    free_response(&res);
    return v;
}

char* voicevox_synthesizer_create_metas_json(const struct VoicevoxSynthesizer* synthesizer) {
    uint64_t s[1] = {(uint64_t)(uintptr_t)synthesizer};
    VvResponse res;
    if (!call_simple(VV_OP_SYN_METAS_JSON, s, 1, &res)) return NULL;
    char* json = NULL;
    if (res.nblob) {
        json = malloc((size_t)res.len[0] + 1);
        if (json) {
            memcpy(json, res.blob[0], res.len[0]);
            json[res.len[0]] = 0;
        }
    }
    free_response(&res);
    return json;
}

VoicevoxResultCode voicevox_synthesizer_create_audio_query_from_kana(
    const struct VoicevoxSynthesizer* synthesizer, const char* kana, VoicevoxStyleId style_id,
    char** output_audio_query_json) {
    return json_from_text(VV_OP_SYN_AUDIO_QUERY_FROM_KANA, synthesizer, kana, style_id,
                          output_audio_query_json);
}

VoicevoxResultCode voicevox_synthesizer_create_audio_query(
    const struct VoicevoxSynthesizer* synthesizer, const char* text, VoicevoxStyleId style_id,
    char** output_audio_query_json) {
    return json_from_text(VV_OP_SYN_AUDIO_QUERY, synthesizer, text, style_id,
                          output_audio_query_json);
}

VoicevoxResultCode voicevox_synthesizer_create_accent_phrases_from_kana(
    const struct VoicevoxSynthesizer* synthesizer, const char* kana, VoicevoxStyleId style_id,
    char** output_accent_phrases_json) {
    return json_from_text(VV_OP_SYN_ACCENT_PHRASES_FROM_KANA, synthesizer, kana, style_id,
                          output_accent_phrases_json);
}

VoicevoxResultCode voicevox_synthesizer_create_accent_phrases(
    const struct VoicevoxSynthesizer* synthesizer, const char* text, VoicevoxStyleId style_id,
    char** output_accent_phrases_json) {
    return json_from_text(VV_OP_SYN_ACCENT_PHRASES, synthesizer, text, style_id,
                          output_accent_phrases_json);
}

VoicevoxResultCode voicevox_synthesizer_replace_mora_data(
    const struct VoicevoxSynthesizer* synthesizer, const char* accent_phrases_json,
    VoicevoxStyleId style_id, char** output_accent_phrases_json) {
    return json_from_text(VV_OP_SYN_REPLACE_MORA_DATA, synthesizer, accent_phrases_json, style_id,
                          output_accent_phrases_json);
}

VoicevoxResultCode voicevox_synthesizer_replace_phoneme_length(
    const struct VoicevoxSynthesizer* synthesizer, const char* accent_phrases_json,
    VoicevoxStyleId style_id, char** output_accent_phrases_json) {
    return json_from_text(VV_OP_SYN_REPLACE_PHONEME_LENGTH, synthesizer, accent_phrases_json,
                          style_id, output_accent_phrases_json);
}

VoicevoxResultCode voicevox_synthesizer_replace_mora_pitch(
    const struct VoicevoxSynthesizer* synthesizer, const char* accent_phrases_json,
    VoicevoxStyleId style_id, char** output_accent_phrases_json) {
    return json_from_text(VV_OP_SYN_REPLACE_MORA_PITCH, synthesizer, accent_phrases_json, style_id,
                          output_accent_phrases_json);
}

VoicevoxResultCode voicevox_synthesizer_synthesis(const struct VoicevoxSynthesizer* synthesizer,
                                                  const char* audio_query_json,
                                                  VoicevoxStyleId style_id,
                                                  struct VoicevoxSynthesisOptions options,
                                                  uintptr_t* output_wav_length,
                                                  uint8_t** output_wav) {
    return wav_call(VV_OP_SYN_SYNTHESIS, synthesizer, audio_query_json, style_id,
                    options.enable_interrogative_upspeak ? 1 : 0, output_wav_length, output_wav);
}

VoicevoxResultCode voicevox_synthesizer_tts_from_kana(
    const struct VoicevoxSynthesizer* synthesizer, const char* kana, VoicevoxStyleId style_id,
    struct VoicevoxTtsOptions options, uintptr_t* output_wav_length, uint8_t** output_wav) {
    return wav_call(VV_OP_SYN_TTS_FROM_KANA, synthesizer, kana, style_id,
                    options.enable_interrogative_upspeak ? 1 : 0, output_wav_length, output_wav);
}

VoicevoxResultCode voicevox_synthesizer_tts(const struct VoicevoxSynthesizer* synthesizer,
                                            const char* text, VoicevoxStyleId style_id,
                                            struct VoicevoxTtsOptions options,
                                            uintptr_t* output_wav_length, uint8_t** output_wav) {
    return wav_call(VV_OP_SYN_TTS, synthesizer, text, style_id,
                    options.enable_interrogative_upspeak ? 1 : 0, output_wav_length, output_wav);
}

VoicevoxResultCode voicevox_synthesizer_frame_synthesis(
    const struct VoicevoxSynthesizer* synthesizer, const char* frame_audio_query_json,
    VoicevoxStyleId style_id, uintptr_t* output_wav_length, uint8_t** output_wav) {
    return wav_call(VV_OP_SYN_FRAME_SYNTHESIS, synthesizer, frame_audio_query_json, style_id, 0,
                    output_wav_length, output_wav);
}

VoicevoxResultCode voicevox_synthesizer_create_sing_frame_audio_query(
    const struct VoicevoxSynthesizer* synthesizer, const char* score_json,
    VoicevoxStyleId style_id, char** output_frame_audio_query_json) {
    return json_from_text(VV_OP_SYN_SING_FRAME_AUDIO_QUERY, synthesizer, score_json, style_id,
                          output_frame_audio_query_json);
}

// The two that take a score *and* a frame audio query.
static VoicevoxResultCode sing_two_json(uint32_t op, const struct VoicevoxSynthesizer* syn,
                                        const char* score_json, const char* frame_json,
                                        VoicevoxStyleId style_id, char** out_json) {
    if (out_json) *out_json = NULL;
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = op;
    req.s[0] = (uint64_t)(uintptr_t)syn;
    req.s[1] = style_id;
    req.nblob = 2;
    req.blob[0] = score_json ? score_json : "";
    req.len[0] = (uint32_t)strlen(score_json ? score_json : "");
    req.blob[1] = frame_json ? frame_json : "";
    req.len[1] = (uint32_t)strlen(frame_json ? frame_json : "");
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    if (res.code == VOICEVOX_RESULT_OK && out_json && res.nblob) {
        *out_json = malloc((size_t)res.len[0] + 1);
        if (*out_json) {
            memcpy(*out_json, res.blob[0], res.len[0]);
            (*out_json)[res.len[0]] = 0;
        }
    }
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_synthesizer_create_sing_frame_f0(
    const struct VoicevoxSynthesizer* synthesizer, const char* score_json,
    const char* frame_audio_query_json, VoicevoxStyleId style_id, char** output_f0_json) {
    return sing_two_json(VV_OP_SYN_SING_FRAME_F0, synthesizer, score_json, frame_audio_query_json,
                         style_id, output_f0_json);
}

VoicevoxResultCode voicevox_synthesizer_create_sing_frame_volume(
    const struct VoicevoxSynthesizer* synthesizer, const char* score_json,
    const char* frame_audio_query_json, VoicevoxStyleId style_id, char** output_volume_json) {
    return sing_two_json(VV_OP_SYN_SING_FRAME_VOLUME, synthesizer, score_json,
                         frame_audio_query_json, style_id, output_volume_json);
}

// These two free host allocations: the guest's copy was released the moment it
// was sent, so there is nothing on the other side to tell about it.
void voicevox_json_free(char* json) { free(json); }
void voicevox_wav_free(uint8_t* wav) { free(wav); }

const char* voicevox_error_result_to_message(VoicevoxResultCode result_code) {
    // The API promises a pointer that stays valid, so each code's message is
    // fetched once and kept.  There are a few dozen codes and they are small.
    enum { CACHE = 64 };
    static const char* cache[CACHE];
    int idx = (result_code >= 0 && result_code < CACHE) ? (int)result_code : -1;
    if (idx >= 0 && cache[idx]) return cache[idx];

    uint64_t s[1] = {(uint64_t)(uint32_t)result_code};
    VvResponse res;
    if (!call_simple(VV_OP_ERROR_MESSAGE, s, 1, &res))
        return "the VOICEVOX emulator backend is not available";
    char* copy = malloc((size_t)res.len[0] + 1);
    if (copy) {
        memcpy(copy, res.blob[0], res.len[0]);
        copy[res.len[0]] = 0;
    }
    free_response(&res);
    if (!copy) return "";
    if (idx >= 0) {
        cache[idx] = copy;
        return cache[idx];
    }
    return copy;  // an out-of-range code is not expected to repeat
}

struct VoicevoxUserDictWord voicevox_user_dict_word_make(const char* surface,
                                                         const char* pronunciation,
                                                         uintptr_t accent_type) {
    // The word_type and priority defaults come from the guest so they cannot
    // drift from the real implementation; the two strings are the caller's own
    // and stay pointing at them, exactly as the original does.
    static int have;
    static VoicevoxUserDictWordType word_type;
    static uint32_t priority;
    if (!have) {
        VvResponse res;
        if (call_simple(VV_OP_DEFAULT_USER_DICT_WORD, NULL, 0, &res)) {
            word_type = (VoicevoxUserDictWordType)(int32_t)(uint32_t)res.r[0];
            priority = (uint32_t)res.r[1];
            have = 1;
            free_response(&res);
        } else {
            word_type = VOICEVOX_USER_DICT_WORD_TYPE_COMMON_NOUN;
            priority = 5;
        }
    }
    struct VoicevoxUserDictWord w;
    w.surface = surface;
    w.pronunciation = pronunciation;
    w.accent_type = accent_type;
    w.word_type = word_type;
    w.priority = priority;
    return w;
}

struct VoicevoxUserDict* voicevox_user_dict_new(void) {
    VvResponse res;
    if (!call_simple(VV_OP_USER_DICT_NEW, NULL, 0, &res)) return NULL;
    struct VoicevoxUserDict* p = (struct VoicevoxUserDict*)(uintptr_t)res.r[0];
    free_response(&res);
    return p;
}

VoicevoxResultCode voicevox_user_dict_load(const struct VoicevoxUserDict* user_dict,
                                           const char* dict_path) {
    char buf[2048];
    const char* p = to_guest_path(dict_path, buf, sizeof buf);
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = VV_OP_USER_DICT_LOAD;
    req.s[0] = (uint64_t)(uintptr_t)user_dict;
    req.nblob = 1;
    req.blob[0] = p ? p : "";
    req.len[0] = (uint32_t)strlen(p ? p : "");
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_user_dict_add_word(const struct VoicevoxUserDict* user_dict,
                                               const struct VoicevoxUserDictWord* word,
                                               uint8_t (*output_word_uuid)[16]) {
    if (output_word_uuid) memset(*output_word_uuid, 0, 16);
    if (!word) return VOICEVOX_RESULT_INVALID_USER_DICT_WORD_ERROR;
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = VV_OP_USER_DICT_ADD_WORD;
    req.s[0] = (uint64_t)(uintptr_t)user_dict;
    req.s[1] = (uint64_t)word->accent_type;
    req.s[2] = (uint64_t)(uint32_t)word->word_type;
    req.s[3] = word->priority;
    req.nblob = 2;
    req.blob[0] = word->surface ? word->surface : "";
    req.len[0] = (uint32_t)strlen(word->surface ? word->surface : "");
    req.blob[1] = word->pronunciation ? word->pronunciation : "";
    req.len[1] = (uint32_t)strlen(word->pronunciation ? word->pronunciation : "");
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    if (res.code == VOICEVOX_RESULT_OK && output_word_uuid && res.nblob && res.len[0] == 16)
        memcpy(*output_word_uuid, res.blob[0], 16);
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_user_dict_update_word(const struct VoicevoxUserDict* user_dict,
                                                  const uint8_t (*word_uuid)[16],
                                                  const struct VoicevoxUserDictWord* word) {
    if (!word || !word_uuid) return VOICEVOX_RESULT_INVALID_USER_DICT_WORD_ERROR;
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = VV_OP_USER_DICT_UPDATE_WORD;
    req.s[0] = (uint64_t)(uintptr_t)user_dict;
    req.s[1] = (uint64_t)word->accent_type;
    req.s[2] = (uint64_t)(uint32_t)word->word_type;
    req.s[3] = word->priority;
    req.nblob = 3;
    req.blob[0] = word->surface ? word->surface : "";
    req.len[0] = (uint32_t)strlen(word->surface ? word->surface : "");
    req.blob[1] = word->pronunciation ? word->pronunciation : "";
    req.len[1] = (uint32_t)strlen(word->pronunciation ? word->pronunciation : "");
    req.blob[2] = *word_uuid;
    req.len[2] = 16;
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_user_dict_remove_word(const struct VoicevoxUserDict* user_dict,
                                                  const uint8_t (*word_uuid)[16]) {
    if (!word_uuid) return VOICEVOX_RESULT_INVALID_UUID_ERROR;
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = VV_OP_USER_DICT_REMOVE_WORD;
    req.s[0] = (uint64_t)(uintptr_t)user_dict;
    req.nblob = 1;
    req.blob[0] = *word_uuid;
    req.len[0] = 16;
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_user_dict_to_json(const struct VoicevoxUserDict* user_dict,
                                              char** output_json) {
    if (output_json) *output_json = NULL;
    uint64_t s[1] = {(uint64_t)(uintptr_t)user_dict};
    VvResponse res;
    if (!call_simple(VV_OP_USER_DICT_TO_JSON, s, 1, &res)) return VV_UNREACHABLE;
    if (res.code == VOICEVOX_RESULT_OK && output_json && res.nblob) {
        *output_json = malloc((size_t)res.len[0] + 1);
        if (*output_json) {
            memcpy(*output_json, res.blob[0], res.len[0]);
            (*output_json)[res.len[0]] = 0;
        }
    }
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_user_dict_import(const struct VoicevoxUserDict* user_dict,
                                             const struct VoicevoxUserDict* other_dict) {
    uint64_t s[2] = {(uint64_t)(uintptr_t)user_dict, (uint64_t)(uintptr_t)other_dict};
    VvResponse res;
    if (!call_simple(VV_OP_USER_DICT_IMPORT, s, 2, &res)) return VV_UNREACHABLE;
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

VoicevoxResultCode voicevox_user_dict_save(const struct VoicevoxUserDict* user_dict,
                                           const char* path) {
    char buf[2048];
    const char* p = to_guest_path(path, buf, sizeof buf);
    VvRequest req;
    memset(&req, 0, sizeof req);
    req.op = VV_OP_USER_DICT_SAVE;
    req.s[0] = (uint64_t)(uintptr_t)user_dict;
    req.nblob = 1;
    req.blob[0] = p ? p : "";
    req.len[0] = (uint32_t)strlen(p ? p : "");
    VvResponse res;
    if (!rpc(&req, &res)) return VV_UNREACHABLE;
    VoicevoxResultCode code = res.code;
    free_response(&res);
    return code;
}

void voicevox_user_dict_delete(struct VoicevoxUserDict* user_dict) {
    uint64_t s[1] = {(uint64_t)(uintptr_t)user_dict};
    VvResponse res;
    if (call_simple(VV_OP_USER_DICT_DELETE, s, 1, &res)) free_response(&res);
}
