// The host half of the shim: what the guest's trampolines reach.
//
// Compiled into the emulator (see src/vvcudaemu.cpp) and installed as
// `Emulator::on_host_call`.  The guest side is src/cudaguest.c; the numbering
// they share is src/vvhostcall.h.
//
// The arithmetic itself is the same code the native shim uses - cudnn_real.cpp,
// cublas_real.cpp, cudakernels.c, unchanged.  Everything here is the boundary:
// deciding which side of it a pointer is on, and copying the small things
// across.  The large things never cross, which is the point.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "emulator.h"
#include "vvhostcall.h"

// The compute side, which knows nothing about any of this.
extern "C" {
int vvstub_run_kernel(const char* name, void** args);
int vvstub_kernel_nargs(const char* name);

int cudnnCreateTensorDescriptor(void** d);
int cudnnDestroyTensorDescriptor(void* d);
int cudnnSetTensorNdDescriptor(void* d, int type, int nb, const int* dim,
                               const int* stride);
int cudnnSetTensor4dDescriptor(void* d, int format, int type, int n, int c, int h,
                               int w);
int cudnnCreateFilterDescriptor(void** d);
int cudnnDestroyFilterDescriptor(void* d);
int cudnnSetFilterNdDescriptor(void* d, int type, int format, int nb, const int* dim);
int cudnnCreateConvolutionDescriptor(void** d);
int cudnnDestroyConvolutionDescriptor(void* d);
int cudnnSetConvolutionNdDescriptor(void* d, int nb, const int* pad, const int* stride,
                                    const int* dilation, int mode, int computeType);
int cudnnSetConvolutionGroupCount(void* d, int groups);
int cudnnConvolutionForward(void* h, const void* alpha, void* xDesc, const void* x,
                            void* wDesc, const void* w, void* convDesc, int algo,
                            void* ws, size_t wsz, const void* beta, void* yDesc,
                            void* y);
int cudnnConvolutionBackwardData(void* h, const void* alpha, void* wDesc,
                                 const void* w, void* dyDesc, const void* dy,
                                 void* convDesc, int algo, void* ws, size_t wsz,
                                 const void* beta, void* dxDesc, void* dx);
int cudnnAddTensor(void* h, const void* alpha, void* aDesc, const void* A,
                   const void* beta, void* cDesc, void* C);
int cublasSgemm_v2(void* h, int transa, int transb, int m, int n, int k,
                   const float* alpha, const float* A, int lda, const float* B, int ldb,
                   const float* beta, float* C, int ldc);
int cublasSgemmStridedBatched(void* h, int transa, int transb, int m, int n, int k,
                              const float* alpha, const float* A, int lda,
                              long long strideA, const float* B, int ldb,
                              long long strideB, const float* beta, float* C, int ldc,
                              long long strideC, int batch);
int cublasSgeam(void* h, int transa, int transb, int m, int n, const float* alpha,
                const float* A, int lda, const float* beta, const float* B, int ldb,
                float* C, int ldc);

// The compute side expects these from cudastub.c, which is the *guest* build.
int vvstub_trace = 0;
int vvstub_timing = 0;
void vvstub_note(const char* name) { std::fprintf(stderr, "[cuda] %s\n", name); }
double vvstub_now(void) { return 0.0; }
void vvstub_account(int bucket, double started) { (void)bucket; (void)started; }
}

namespace {

// ---------------------------------------------------------------------------
// Device memory
//
// One arena, at an address no guest pointer can be mistaken for.  That matters:
// a guest address and a host address are both just 64-bit numbers, and the
// emulator's guest layout (0x5555..., 0x7fff...) is exactly where Linux puts a
// real heap.  Classifying by "is it inside my arena" is unambiguous where
// classifying by "did I allocate it" would not be - ONNX Runtime sub-allocates
// inside a cudaMalloc'd block, so the pointers it passes back are rarely the
// ones it was given.

constexpr uint64_t kArenaBase = 0x0000'3000'0000'0000ull;
constexpr uint64_t kArenaSize = 4ull << 30;  // reserved, not committed

uint8_t* arena_start;
uint64_t arena_used;
// Freed blocks, by size, for reuse: ONNX Runtime's own arena allocates a few
// large blocks and recycles them, so best fit here is nearly always exact.
std::multimap<uint64_t, uint64_t> arena_free;  // size -> offset

bool arena_init() {
    if (arena_start) return true;
#if defined(_WIN32)
    arena_start = static_cast<uint8_t*>(VirtualAlloc(reinterpret_cast<void*>(kArenaBase),
                                                     kArenaSize, MEM_RESERVE,
                                                     PAGE_READWRITE));
#else
    void* p = mmap(reinterpret_cast<void*>(kArenaBase), kArenaSize,
                   PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                   -1, 0);
    arena_start = (p == MAP_FAILED) ? nullptr : static_cast<uint8_t*>(p);
#endif
    if (!arena_start) {
        std::fprintf(stderr, "[cuda] cannot reserve the device arena\n");
        return false;
    }
    return true;
}

bool is_device(uint64_t p) {
    if (!arena_start || !p) return false;
    uint64_t base = reinterpret_cast<uint64_t>(arena_start);
    return p >= base && p < base + kArenaSize;
}

uint64_t device_alloc(uint64_t n) {
    if (!arena_init()) return 0;
    n = (n + 255) & ~255ull;  // ORT wants its tensors aligned; 256 is generous
    auto it = arena_free.lower_bound(n);
    if (it != arena_free.end() && it->first <= n * 2) {
        uint64_t off = it->second;
        arena_free.erase(it);
        return reinterpret_cast<uint64_t>(arena_start) + off;
    }
    if (arena_used + n > kArenaSize) return 0;
    uint64_t off = arena_used;
    arena_used += n;
#if defined(_WIN32)
    if (!VirtualAlloc(arena_start + off, n, MEM_COMMIT, PAGE_READWRITE)) return 0;
#endif
    return reinterpret_cast<uint64_t>(arena_start) + off;
}

// Sizes are remembered so that a freed block can go back on the list.
std::map<uint64_t, uint64_t> live_blocks;  // offset -> size

void device_free(uint64_t p) {
    if (!is_device(p)) return;
    uint64_t off = p - reinterpret_cast<uint64_t>(arena_start);
    auto it = live_blocks.find(off);
    if (it == live_blocks.end()) return;
    arena_free.emplace(it->second, off);
    live_blocks.erase(it);
}

// ---------------------------------------------------------------------------
// Reading the small things out of guest memory
//
// Kernel arguments, descriptor dimensions, alpha and beta: all of them live on
// ONNX Runtime's own stack, which is guest memory.  None is larger than a few
// hundred bytes and they are the only bytes that cross.

std::vector<uint8_t> guest_bytes(x86emu::Emulator& e, uint64_t addr, uint64_t len) {
    std::vector<uint8_t> out(len);
    if (len) e.mem.read(addr, out.data(), len);
    return out;
}

// How much of an argument to copy.  The kernels take pointers (8), DivMods
// (12), TArray<int64_t,8> (72), TArray<DivMod,8> (100) and TArray<void*,32>
// (264) - so 264 covers everything, and the handler reads only what its
// signature says.  Copying more than the guest allocated is harmless *unless*
// it runs off the end of a mapping, which is why this backs off on a fault
// rather than assuming.
uint64_t guest_read_tolerant(x86emu::Emulator& e, uint64_t addr, uint8_t* dst,
                             uint64_t want) {
    while (want) {
        try {
            e.mem.read(addr, dst, want);
            return want;
        } catch (const x86emu::MemoryFault&) {
            want /= 2;
        }
    }
    return 0;
}

float guest_float(x86emu::Emulator& e, uint64_t addr, float fallback) {
    if (!addr) return fallback;
    uint32_t bits = e.mem.read32(addr);
    float v;
    std::memcpy(&v, &bits, sizeof v);
    return v;
}

std::vector<int> guest_ints(x86emu::Emulator& e, uint64_t addr, int n) {
    std::vector<int> out(n > 0 ? n : 0);
    for (int i = 0; i < n; i++) out[i] = static_cast<int>(e.mem.read32(addr + 4u * i));
    return out;
}

// ---------------------------------------------------------------------------

int64_t do_memcpy(x86emu::Emulator& e, uint64_t dst, uint64_t src, uint64_t n) {
    if (!n) return 0;
    bool dd = is_device(dst), sd = is_device(src);
    if (dd && sd) {
        std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), n);
    } else if (dd) {
        e.mem.read(src, reinterpret_cast<void*>(dst), n);      // guest -> device
    } else if (sd) {
        e.mem.write(dst, reinterpret_cast<const void*>(src), n);  // device -> guest
    } else {
        std::vector<uint8_t> tmp(n);
        e.mem.read(src, tmp.data(), n);
        e.mem.write(dst, tmp.data(), n);
    }
    return 0;
}

int64_t do_launch(x86emu::Emulator& e, uint64_t name_addr, uint64_t args_addr) {
    std::string name = e.mem.read_cstring(name_addr);
    int nargs = vvstub_kernel_nargs(name.c_str());
    if (nargs <= 0) return 0;  // not one this build knows

    // Each slot is a guest pointer to the argument's value.  The values come
    // over; the tensors they point at do not.
    constexpr uint64_t kMaxArg = 288;
    static std::vector<uint8_t> storage;
    storage.assign(static_cast<size_t>(nargs) * kMaxArg, 0);
    std::vector<void*> host_args(static_cast<size_t>(nargs));
    for (int i = 0; i < nargs; i++) {
        uint64_t p = e.mem.read64(args_addr + 8u * i);
        uint8_t* slot = storage.data() + static_cast<size_t>(i) * kMaxArg;
        if (p) guest_read_tolerant(e, p, slot, kMaxArg);
        host_args[static_cast<size_t>(i)] = slot;
    }
    return vvstub_run_kernel(name.c_str(), host_args.data()) ? 1 : 0;
}

}  // namespace

// ---------------------------------------------------------------------------

int64_t vv_host_call(x86emu::Emulator& e, uint64_t id, uint64_t argp) {
    uint64_t a[VVHOST_SLOTS];
    for (int i = 0; i < VVHOST_SLOTS; i++) a[i] = e.mem.read64(argp + 8u * i);

    auto ptr = [](uint64_t v) { return reinterpret_cast<void*>(v); };
    auto cptr = [](uint64_t v) { return reinterpret_cast<const void*>(v); };

    switch (id) {
        case VVH_ALIVE:
            return 1;

        case VVH_MALLOC: {
            uint64_t p = device_alloc(a[0]);
            if (p) live_blocks[p - reinterpret_cast<uint64_t>(arena_start)] =
                (a[0] + 255) & ~255ull;
            return static_cast<int64_t>(p);
        }
        case VVH_FREE:
            device_free(a[0]);
            return 0;

        case VVH_MEMCPY:
            return do_memcpy(e, a[0], a[1], a[2]);

        case VVH_MEMCPY2D: {
            // (dst, dpitch, src, spitch, width, height)
            for (uint64_t r = 0; r < a[5]; r++)
                do_memcpy(e, a[0] + r * a[1], a[2] + r * a[3], a[4]);
            return 0;
        }
        case VVH_MEMSET: {
            if (is_device(a[0]))
                std::memset(ptr(a[0]), static_cast<int>(a[1]), a[2]);
            else {
                std::vector<uint8_t> tmp(a[2], static_cast<uint8_t>(a[1]));
                if (a[2]) e.mem.write(a[0], tmp.data(), a[2]);
            }
            return 0;
        }
        case VVH_LAUNCH:
            return do_launch(e, a[0], a[1]);

        // ---- cuDNN: descriptors live here, the guest holds only handles -----
        case VVH_CUDNN_CREATE_TENSOR: {
            void* d = nullptr;
            cudnnCreateTensorDescriptor(&d);
            return reinterpret_cast<int64_t>(d);
        }
        case VVH_CUDNN_DESTROY_TENSOR:
            return cudnnDestroyTensorDescriptor(ptr(a[0]));
        case VVH_CUDNN_CREATE_FILTER: {
            void* d = nullptr;
            cudnnCreateFilterDescriptor(&d);
            return reinterpret_cast<int64_t>(d);
        }
        case VVH_CUDNN_DESTROY_FILTER:
            return cudnnDestroyFilterDescriptor(ptr(a[0]));
        case VVH_CUDNN_CREATE_CONV: {
            void* d = nullptr;
            cudnnCreateConvolutionDescriptor(&d);
            return reinterpret_cast<int64_t>(d);
        }
        case VVH_CUDNN_DESTROY_CONV:
            return cudnnDestroyConvolutionDescriptor(ptr(a[0]));

        case VVH_CUDNN_SET_TENSOR_ND: {
            int nb = static_cast<int>(a[2]);
            std::vector<int> dim = guest_ints(e, a[3], nb);
            std::vector<int> stride = a[4] ? guest_ints(e, a[4], nb) : std::vector<int>();
            return cudnnSetTensorNdDescriptor(ptr(a[0]), static_cast<int>(a[1]), nb,
                                              dim.data(),
                                              stride.empty() ? nullptr : stride.data());
        }
        case VVH_CUDNN_SET_TENSOR_4D:
            return cudnnSetTensor4dDescriptor(
                ptr(a[0]), static_cast<int>(a[1]), static_cast<int>(a[2]),
                static_cast<int>(a[3]), static_cast<int>(a[4]), static_cast<int>(a[5]),
                static_cast<int>(a[6]));
        case VVH_CUDNN_SET_FILTER_ND: {
            int nb = static_cast<int>(a[3]);
            std::vector<int> dim = guest_ints(e, a[4], nb);
            return cudnnSetFilterNdDescriptor(ptr(a[0]), static_cast<int>(a[1]),
                                              static_cast<int>(a[2]), nb, dim.data());
        }
        case VVH_CUDNN_SET_CONV_ND: {
            int nb = static_cast<int>(a[1]);
            std::vector<int> pad = guest_ints(e, a[2], nb);
            std::vector<int> stride = guest_ints(e, a[3], nb);
            std::vector<int> dil = guest_ints(e, a[4], nb);
            return cudnnSetConvolutionNdDescriptor(ptr(a[0]), nb, pad.data(),
                                                   stride.data(), dil.data(),
                                                   static_cast<int>(a[5]),
                                                   static_cast<int>(a[6]));
        }
        case VVH_CUDNN_SET_CONV_GROUPS:
            return cudnnSetConvolutionGroupCount(ptr(a[0]), static_cast<int>(a[1]));

        case VVH_CUDNN_CONV_FORWARD: {
            float alpha = guest_float(e, a[0], 1.0f);
            float beta = guest_float(e, a[7], 0.0f);
            return cudnnConvolutionForward(nullptr, &alpha, ptr(a[1]), cptr(a[2]),
                                           ptr(a[3]), cptr(a[4]), ptr(a[5]),
                                           static_cast<int>(a[6]), nullptr, 0, &beta,
                                           ptr(a[8]), ptr(a[9]));
        }
        case VVH_CUDNN_CONV_BACKWARD_DATA: {
            float alpha = guest_float(e, a[0], 1.0f);
            float beta = guest_float(e, a[7], 0.0f);
            return cudnnConvolutionBackwardData(nullptr, &alpha, ptr(a[1]), cptr(a[2]),
                                                ptr(a[3]), cptr(a[4]), ptr(a[5]),
                                                static_cast<int>(a[6]), nullptr, 0,
                                                &beta, ptr(a[8]), ptr(a[9]));
        }
        case VVH_CUDNN_ADD_TENSOR: {
            float alpha = guest_float(e, a[0], 1.0f);
            float beta = guest_float(e, a[3], 0.0f);
            return cudnnAddTensor(nullptr, &alpha, ptr(a[1]), cptr(a[2]), &beta,
                                  ptr(a[4]), ptr(a[5]));
        }

        // ---- cuBLAS ---------------------------------------------------------
        case VVH_CUBLAS_SGEMM: {
            float alpha = guest_float(e, a[5], 1.0f);
            float beta = guest_float(e, a[10], 0.0f);
            return cublasSgemm_v2(
                nullptr, static_cast<int>(a[0]), static_cast<int>(a[1]),
                static_cast<int>(a[2]), static_cast<int>(a[3]), static_cast<int>(a[4]),
                &alpha, static_cast<const float*>(cptr(a[6])), static_cast<int>(a[7]),
                static_cast<const float*>(cptr(a[8])), static_cast<int>(a[9]), &beta,
                static_cast<float*>(ptr(a[11])), static_cast<int>(a[12]));
        }
        case VVH_CUBLAS_SGEMM_BATCHED: {
            float alpha = guest_float(e, a[5], 1.0f);
            float beta = guest_float(e, a[10], 0.0f);
            // The last slot points at {strideA, strideB, strideC, batch} on the
            // guest's stack: seventeen arguments would not otherwise fit.
            long long extra[4] = {0, 0, 0, 0};
            for (int i = 0; i < 4; i++)
                extra[i] = static_cast<long long>(e.mem.read64(a[13] + 8u * i));
            return cublasSgemmStridedBatched(
                nullptr, static_cast<int>(a[0]), static_cast<int>(a[1]),
                static_cast<int>(a[2]), static_cast<int>(a[3]), static_cast<int>(a[4]),
                &alpha, static_cast<const float*>(cptr(a[6])), static_cast<int>(a[7]),
                extra[0], static_cast<const float*>(cptr(a[8])), static_cast<int>(a[9]),
                extra[1], &beta, static_cast<float*>(ptr(a[11])),
                static_cast<int>(a[12]), extra[2], static_cast<int>(extra[3]));
        }
        case VVH_CUBLAS_SGEAM: {
            float alpha = guest_float(e, a[4], 1.0f);
            float beta = guest_float(e, a[7], 0.0f);
            return cublasSgeam(nullptr, static_cast<int>(a[0]), static_cast<int>(a[1]),
                               static_cast<int>(a[2]), static_cast<int>(a[3]), &alpha,
                               static_cast<const float*>(cptr(a[5])),
                               static_cast<int>(a[6]), &beta,
                               static_cast<const float*>(cptr(a[8])),
                               static_cast<int>(a[9]),
                               static_cast<float*>(ptr(a[10])), static_cast<int>(a[11]));
        }

        default:
            std::fprintf(stderr, "[cuda] host call %llu is not implemented\n",
                         static_cast<unsigned long long>(id));
            return -1;
    }
}
