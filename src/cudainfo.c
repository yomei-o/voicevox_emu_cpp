// The parts of the CUDA runtime that only have to answer plausibly.
//
// Shared by both builds of the stand-ins - the native one (src/cudastub.c) and
// the emulated one (src/cudaguest.c) - because none of it depends on where the
// arithmetic happens.  Registration bookkeeping, a device that looks like a
// Turing card, streams that only have to be distinct.
//
// The one thing that is not bookkeeping is `__cudaRegisterFunction`, which is
// where a kernel's *name* comes from; that lives with whichever build owns the
// launch path.
#include "cudastub.h"

#include <string.h>

// ---- registration ----------------------------------------------------------
// The fat binary is CUDA's container for device code.  None of this is public
// API - it is what nvcc emits calls to - but it is stable enough that every
// CUDA binary since 9.0 uses these.  The handle is only ever handed back.

void** __cudaRegisterFatBinary(void* fat) {
    static void* handle;
    handle = fat;
    return &handle;
}

void __cudaRegisterFatBinaryEnd(void** handle) { (void)handle; }
void __cudaUnregisterFatBinary(void** handle) { (void)handle; }

void __cudaRegisterVar(void** handle, char* host_var, char* device_addr,
                       const char* device_name, int ext, size_t size, int constant,
                       int global) {
    (void)handle; (void)host_var; (void)device_addr; (void)device_name;
    (void)ext; (void)size; (void)constant; (void)global;
}

// The launch configuration is pushed before a `<<<>>>` call and popped inside
// the generated stub.  Keeping one frame is enough: the provider is
// single-threaded through this path.
static struct {
    unsigned long long grid[3], block[3];
    size_t shared;
    void* stream;
} pending;

int __cudaPushCallConfiguration(unsigned long long gx, unsigned long long gy,
                                unsigned long long bx, unsigned long long by,
                                size_t shared, void* stream) {
    // The x/y halves of dim3 arrive packed two per register.
    pending.grid[0] = gx;
    pending.grid[1] = gy;
    pending.block[0] = bx;
    pending.block[1] = by;
    pending.shared = shared;
    pending.stream = stream;
    return 0;
}

int __cudaPopCallConfiguration(void* gridDim, void* blockDim, size_t* shared,
                               void* stream) {
    if (gridDim) memcpy(gridDim, pending.grid, sizeof pending.grid[0] * 2);
    if (blockDim) memcpy(blockDim, pending.block, sizeof pending.block[0] * 2);
    if (shared) *shared = pending.shared;
    if (stream) memcpy(stream, &pending.stream, sizeof pending.stream);
    return 0;
}

// ---- the device, which has to look plausible --------------------------------

int cudaMemGetInfo(size_t* free_, size_t* total) {
    if (total) *total = (size_t)8 << 30;
    if (free_) *free_ = (size_t)6 << 30;
    return 0;
}

int cudaGetDeviceCount(int* n) { *n = 1; return 0; }
int cudaGetDevice(int* d) { *d = 0; return 0; }
int cudaSetDevice(int d) { (void)d; return 0; }
int cudaRuntimeGetVersion(int* v) { *v = 12000; return 0; }

// cudaDeviceProp is large and its layout is version-specific; zeroing it and
// filling only what a provider is likely to read is the safe move.
int cudaGetDeviceProperties_v2(void* prop, int device) {
    (void)device;
    if (!prop) return 0;
    memset(prop, 0, 1024);
    strcpy((char*)prop, "vvstub");   // name[256]
    return 0;
}

int cudaDeviceGetAttribute(int* value, int attr, int device) {
    (void)device;
    // Answers chosen so that occupancy and block-size maths stay sane.
    switch (attr) {
        case 1: *value = 1024; break;    // MaxThreadsPerBlock
        case 16: *value = 80; break;     // MultiProcessorCount
        case 36: *value = 7; break;      // ComputeCapabilityMajor
        case 37: *value = 5; break;      // ComputeCapabilityMinor
        default: *value = 1; break;
    }
    return 0;
}

static const char* kOk = "vvstub: no error";
const char* cudaGetErrorString(int e) { (void)e; return kOk; }
const char* cudaGetErrorName(int e) { (void)e; return kOk; }

// ---- streams, which only have to be distinct --------------------------------

int cudaStreamCreateWithFlags(void** s, unsigned flags) {
    (void)flags;
    static long counter;
    *s = (void*)(++counter);
    return 0;
}
int cudaStreamDestroy(void* s) { (void)s; return 0; }
int cudaStreamSynchronize(void* s) { (void)s; return 0; }
