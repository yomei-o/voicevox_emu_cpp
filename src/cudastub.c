// The handful of CUDA entry points that have to do something real.
//
// Everything else in the generated stand-ins returns success and computes
// nothing.  These do not, because without them ONNX Runtime never gets far
// enough to launch a kernel, which is the whole point of the exercise:
//
//   - the allocator hands out ordinary host memory, so "device" pointers are
//     real pointers and the copies below can be memcpy;
//   - the device query answers with a plausible GPU, so the provider does not
//     decline to initialise;
//   - `__cudaRegisterFunction` records every kernel's name against its host
//     stub address, which is exactly the mapping `cudaLaunchKernel` needs to
//     say *which* kernel was launched.
//
// Nothing here computes a result.  A launched kernel prints its name and
// returns, so the numbers coming out the other end are meaningless - the list
// of names is the output.
#include "cudastub.h"

#include <string.h>
#include <time.h>

int vvstub_trace = 0;
int vvstub_timing = 0;

// The three stand-in libraries are separate objects, but only libcudart links
// this file; the others resolve these at load time against it.
static double bucket_seconds[VVSTUB_T_COUNT];
static long long bucket_calls[VVSTUB_T_COUNT];

double vvstub_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

void vvstub_account(int bucket, double started) {
    if (bucket < 0 || bucket >= VVSTUB_T_COUNT) return;
    bucket_seconds[bucket] += vvstub_now() - started;
    bucket_calls[bucket]++;
}

__attribute__((constructor)) static void vvstub_init(void) {
    const char* t = getenv("VVSTUB_TRACE");
    vvstub_trace = t && *t && *t != '0';
    const char* m = getenv("VVSTUB_TIME");
    vvstub_timing = m && *m && *m != '0';
}

void vvstub_note(const char* name) {
    fprintf(stderr, "[cuda] %s\n", name);
    fflush(stderr);
}

// ---- the kernel registry --------------------------------------------------
// A CUDA binary registers each kernel at load time, handing the runtime the
// address of a host-side stub and the device symbol's name.  Keeping that pair
// is what turns an opaque function pointer at launch time into a name.

typedef struct {
    const void* host_fn;
    const char* name;
} Kernel;

static Kernel* kernels;
static size_t kernel_count, kernel_cap;

static void remember(const void* host_fn, const char* name) {
    if (kernel_count == kernel_cap) {
        kernel_cap = kernel_cap ? kernel_cap * 2 : 1024;
        kernels = realloc(kernels, kernel_cap * sizeof *kernels);
        if (!kernels) abort();
    }
    kernels[kernel_count].host_fn = host_fn;
    kernels[kernel_count].name = name;
    kernel_count++;
}

static const char* name_of(const void* host_fn) {
    for (size_t i = 0; i < kernel_count; i++)
        if (kernels[i].host_fn == host_fn) return kernels[i].name;
    return "(unregistered)";
}

// The registration entry points.  Their signatures are not public API, but
// they are stable enough that every CUDA binary since 9.0 uses these.
void __cudaRegisterFunction(void** handle, const char* host_fn, char* device_fn,
                            const char* device_name, int thread_limit, void* tid,
                            void* bid, void* bDim, void* gDim, int* wSize) {
    (void)handle; (void)device_fn; (void)thread_limit;
    (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;
    remember(host_fn, device_name);
}

// ---- what the experiment is for -------------------------------------------

static size_t launches, unhandled;

int cudaLaunchKernel(const void* func, unsigned long long gx, unsigned long long gy,
                     unsigned long long bx, unsigned long long by, void** args,
                     size_t shared, void* stream) {
    (void)gx; (void)gy; (void)bx; (void)by; (void)shared; (void)stream;
    launches++;
    double t0 = vvstub_timing ? vvstub_now() : 0;
    const char* name = name_of(func);
    // A kernel this build knows how to do natively gets done; the rest are
    // still counted and named, which is what the enumeration needed.
    int handled = vvstub_run_kernel(name, args);
    if (vvstub_timing) vvstub_account(VVSTUB_T_KERNEL, t0);
    if (!handled) unhandled++;
    if (vvstub_trace || !handled) {
        printf("kernel %s%s\n", handled ? "[done] " : "", name);
        fflush(stdout);
    }
    return 0;
}

__attribute__((destructor)) static void vvstub_report(void) {
    fprintf(stderr, "[cuda] %zu kernels registered, %zu launches\n", kernel_count,
            launches);
    if (vvstub_timing) {
        static const char* names[VVSTUB_T_COUNT] = {"kernels", "cuDNN", "cuBLAS"};
        double total = 0;
        for (int i = 0; i < VVSTUB_T_COUNT; i++) {
            total += bucket_seconds[i];
            fprintf(stderr, "[time] %-8s %8.3f s  %lld calls\n", names[i],
                    bucket_seconds[i], bucket_calls[i]);
        }
        fprintf(stderr,
                "[time] %-8s %8.3f s  <- the arithmetic; whatever the caller "
                "measured beyond this is ONNX Runtime's own\n",
                "total", total);
    }
    fflush(stderr);
}

// ---- memory, which has to be real for anything to proceed -----------------

int cudaMalloc(void** p, size_t n) {
    *p = malloc(n ? n : 1);
    return *p ? 0 : 2;  // cudaErrorMemoryAllocation
}
int cudaFree(void* p) { free(p); return 0; }
int cudaMallocHost(void** p, size_t n) { return cudaMalloc(p, n); }
int cudaHostAlloc(void** p, size_t n, unsigned flags) { (void)flags; return cudaMalloc(p, n); }
int cudaFreeHost(void* p) { free(p); return 0; }

int cudaMemcpy(void* dst, const void* src, size_t n, int kind) {
    (void)kind;
    memcpy(dst, src, n);
    return 0;
}
int cudaMemcpyAsync(void* dst, const void* src, size_t n, int kind, void* stream) {
    (void)stream;
    return cudaMemcpy(dst, src, n, kind);
}
int cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                      size_t width, size_t height, int kind, void* stream) {
    (void)kind; (void)stream;
    for (size_t r = 0; r < height; r++)
        memcpy((char*)dst + r * dpitch, (const char*)src + r * spitch, width);
    return 0;
}
int cudaMemset(void* p, int v, size_t n) { memset(p, v, n); return 0; }
int cudaMemsetAsync(void* p, int v, size_t n, void* stream) {
    (void)stream;
    return cudaMemset(p, v, n);
}
