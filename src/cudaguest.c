// The guest half of the shim: trampolines that hand the work to the host.
//
// This replaces src/cudastub.c + src/cudakernels.c + the two _real.cpp files
// when the stand-ins are built for the emulator rather than for a native run.
// Nothing here computes: every entry point that does real work forwards to the
// host through one reserved syscall, and the arithmetic happens at full speed
// outside the interpreter.  See src/vvhostcall.h for why device memory can be
// host memory and the tensors therefore never move.
//
// What stays on this side:
//
//   * the kernel registry.  `__cudaRegisterFunction` fires 4757 times at load;
//     crossing to the host for each would cost more than it saves, and all the
//     host needs at launch time is the name.
//   * pinned host memory, which the guest genuinely reads, so it is guest
//     memory from the guest's own malloc.
//   * every entry point with no behaviour, which is most of them.  A stub that
//     returns success costs one instruction; a syscall costs rather more.
#include "cudastub.h"
#include "vvhostcall.h"

#include <string.h>

int vvstub_trace = 0;
int vvstub_timing = 0;

void vvstub_note(const char* name) {
    fprintf(stderr, "[cuda] %s\n", name);
    fflush(stderr);
}

double vvstub_now(void) { return 0.0; }
void vvstub_account(int bucket, double started) { (void)bucket; (void)started; }

__attribute__((constructor)) static void vvguest_init(void) {
    const char* t = getenv("VVSTUB_TRACE");
    vvstub_trace = t && *t && *t != '0';
    VVA(0);
    if (vvhost(VVH_ALIVE, a) != 1)
        fprintf(stderr,
                "[cuda] no host shim behind this emulator - every kernel will "
                "answer success and compute nothing\n");
}

// ---- the kernel registry ---------------------------------------------------
// A CUDA binary registers each kernel at load time, handing the runtime the
// address of a host-side stub and the device symbol's name.  Keeping that pair
// is what turns an opaque function pointer at launch time into a name, and the
// name is all the host needs.

typedef struct {
    const void* fn;
    const char* name;
} Kernel;

static Kernel* kernels;
static size_t kernel_count, kernel_cap;
static size_t launches, unhandled;

void __cudaRegisterFunction(void** fatbin, const char* host_fn, char* device_fn,
                            const char* name, int tid, void* bid, void* bdim,
                            void* gdim, int* wsize) {
    (void)fatbin; (void)device_fn; (void)tid; (void)bid; (void)bdim; (void)gdim;
    (void)wsize;
    if (kernel_count == kernel_cap) {
        size_t want = kernel_cap ? kernel_cap * 2 : 1024;
        Kernel* grown = realloc(kernels, want * sizeof *grown);
        if (!grown) return;
        kernels = grown;
        kernel_cap = want;
    }
    kernels[kernel_count].fn = host_fn;
    kernels[kernel_count].name = name;
    kernel_count++;
}

static const char* name_of(const void* fn) {
    for (size_t i = 0; i < kernel_count; i++)
        if (kernels[i].fn == fn) return kernels[i].name;
    return "?";
}

int cudaLaunchKernel(const void* func, unsigned long long gx, unsigned long long gy,
                     unsigned long long bx, unsigned long long by, void** args,
                     size_t shared, void* stream) {
    (void)gx; (void)gy; (void)bx; (void)by; (void)shared; (void)stream;
    launches++;
    const char* name = name_of(func);
    VVA((long)name, (long)args);
    long handled = vvhost(VVH_LAUNCH, a);
    if (!handled) unhandled++;
    if (vvstub_trace || !handled) {
        printf("kernel %s%s\n", handled ? "[done] " : "", name);
        fflush(stdout);
    }
    return 0;
}

__attribute__((destructor)) static void vvguest_report(void) {
    fprintf(stderr, "[cuda] %zu kernels registered, %zu launches\n", kernel_count,
            launches);
    // A kernel nothing implements returns success and computes nothing, and the
    // audio that comes out is wrong rather than absent.  That is the failure
    // mode this project spent a day learning to distrust, so it says so - the
    // per-launch lines above scroll away, and a count at the end does not.
    // A model this table was not written against is exactly where it will
    // happen.
    if (unhandled)
        fprintf(stderr,
                "[cuda] WARNING: %zu launches computed nothing - no host "
                "implementation for those kernels, so the output is wrong\n",
                unhandled);
    fflush(stderr);
}

// ---- memory ----------------------------------------------------------------
// Device memory is host memory, which is why none of the tensors ever move.
// Pinned memory is the opposite case: the guest reads it, so it is the guest's
// own malloc and never leaves.

int cudaMalloc(void** p, size_t n) {
    VVA((long)(n ? n : 1));
    long r = vvhost(VVH_MALLOC, a);
    *p = (void*)r;
    return r ? 0 : 2;  // cudaErrorMemoryAllocation
}

int cudaFree(void* p) {
    VVA((long)p);
    if (p) vvhost(VVH_FREE, a);
    return 0;
}

int cudaMallocHost(void** p, size_t n) {
    *p = malloc(n ? n : 1);
    return *p ? 0 : 2;
}
int cudaHostAlloc(void** p, size_t n, unsigned flags) {
    (void)flags;
    return cudaMallocHost(p, n);
}
int cudaFreeHost(void* p) {
    free(p);
    return 0;
}

// The `kind` argument is passed on but not trusted: the host knows which
// pointers it handed out, which is a more reliable answer than a caller's
// cudaMemcpyDefault.
int cudaMemcpy(void* dst, const void* src, size_t n, int kind) {
    VVA((long)dst, (long)src, (long)n, (long)kind);
    return (int)vvhost(VVH_MEMCPY, a);
}
int cudaMemcpyAsync(void* dst, const void* src, size_t n, int kind, void* stream) {
    (void)stream;
    return cudaMemcpy(dst, src, n, kind);
}
int cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                      size_t width, size_t height, int kind, void* stream) {
    (void)stream;
    VVA((long)dst, (long)dpitch, (long)src, (long)spitch, (long)width, (long)height,
        (long)kind);
    return (int)vvhost(VVH_MEMCPY2D, a);
}
int cudaMemset(void* p, int value, size_t n) {
    VVA((long)p, (long)value, (long)n);
    return (int)vvhost(VVH_MEMSET, a);
}
int cudaMemsetAsync(void* p, int value, size_t n, void* stream) {
    (void)stream;
    return cudaMemset(p, value, n);
}
