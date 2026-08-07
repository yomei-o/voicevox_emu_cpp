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
#include <ctime>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "cudastub.h"
#include "emulator.h"
#define VVHOSTCALL_HOST_SIDE 1
#include "vvhostcall.h"

// The compute side, which knows nothing about any of this.
extern "C" {
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

double vvstub_now(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return t.tv_sec + t.tv_nsec / 1e9;
}

// VVSTUB_TIME=1: how the shim's own seconds divide, and - the number that
// matters now - how much of a run is spent on this side of the boundary at all.
// Whatever is left is the guest still being interpreted.
double vvhost_seconds[VVSTUB_T_COUNT + 1];
long long vvhost_calls[VVSTUB_T_COUNT + 1];

void vvstub_account(int bucket, double started) {
    if (bucket < 0 || bucket > VVSTUB_T_COUNT) return;
    vvhost_seconds[bucket] += vvstub_now() - started;
    vvhost_calls[bucket]++;
}
}

namespace {

// ---------------------------------------------------------------------------
// Device memory, in the guest's address space
//
// A CUDA device pointer is never dereferenced by ONNX Runtime - it cannot be,
// that is what "device" means - so it was tempting to make one a *host* pointer
// and skip the guest's address space entirely.  That worked, and cost two
// things that only showed up later:
//
//   a 32-bit host indexing an array of guest pointers strides by four, so the
//   second element is the top half of the first;
//
//   and a saved session cannot be carried between hosts, because host
//   addresses are baked into guest memory and nothing can find them again to
//   rewrite them.
//
// So the arena lives at a guest address, backed by one contiguous host
// allocation (Memory::map_contiguous).  Everything the guest holds is a guest
// address; the shim turns one into a host pointer with a subtraction; and the
// tensors still never move, which was the point of the host arena in the first
// place.

constexpr uint64_t kArenaBase = 0x0000'3000'0000'0000ull;

uint64_t arena_used;
uint64_t arena_size;
uint8_t* arena_host;          // where map_contiguous put it

// Freed blocks, by size, for reuse: ONNX Runtime's own arena allocates a few
// large blocks and recycles them, so best fit here is nearly always exact.
std::multimap<uint64_t, uint64_t> arena_free;  // size -> offset
std::map<uint64_t, uint64_t> live_blocks;      // offset -> size

bool arena_init(x86emu::Emulator& e) {
    if (arena_size) return arena_host != nullptr;
    // Big enough for the weights and the activations: the sessions themselves
    // take 69 MB, and running out is a hard failure rather than a slow one.
    // VVARENA overrides, in megabytes.
    arena_size = 512ull << 20;
    if (const char* mb = std::getenv("VVARENA")) {
        unsigned long long want = std::strtoull(mb, nullptr, 10);
        if (want) arena_size = want << 20;
    }
    e.mem.map_contiguous(kArenaBase, arena_size, "cuda device memory");
    arena_host = e.mem.host_span(kArenaBase, arena_size);
    if (!arena_host) {
        std::fprintf(stderr, "[cuda] cannot map %.0f MB of device memory\n",
                     arena_size / 1048576.0);
        arena_size = 0;
        return false;
    }
    return true;
}

bool is_device(uint64_t p) {
    return arena_size && p >= kArenaBase && p < kArenaBase + arena_size;
}

// The host address of a guest device pointer - the one place the two kinds of
// address meet, and a subtraction.
uint8_t* device_host(uint64_t p) {
    return is_device(p) ? arena_host + (p - kArenaBase) : nullptr;
}

// The same, as whatever the callee wants it to be.  cuBLAS deals in floats and
// cuDNN in void*, and a cast at every call site reads worse than a name.
template <typename T>
T* device_as(uint64_t p) {
    return reinterpret_cast<T*>(device_host(p));
}

uint64_t device_alloc(x86emu::Emulator& e, uint64_t n) {
    if (!arena_init(e)) return 0;
    n = (n + 255) & ~255ull;  // ORT wants its tensors aligned; 256 is generous
    auto it = arena_free.lower_bound(n);
    if (it != arena_free.end() && it->first <= n * 2) {
        // Both fields read before the erase: `it` is not a thing afterwards.
        uint64_t off = it->second, had = it->first;
        arena_free.erase(it);
        live_blocks[off] = had;
        return kArenaBase + off;
    }
    if (arena_used + n > arena_size) {
        std::fprintf(stderr,
                     "[cuda] the device arena is full: %.0f MB used, %.0f MB asked "
                     "for, %.0f MB in all - raise VVARENA\n",
                     arena_used / 1048576.0, n / 1048576.0, arena_size / 1048576.0);
        return 0;
    }
    uint64_t off = arena_used;
    arena_used += n;
    live_blocks[off] = n;
    return kArenaBase + off;
}

void device_free(uint64_t p) {
    if (!is_device(p)) return;
    uint64_t off = p - kArenaBase;
    auto it = live_blocks.find(off);
    if (it == live_blocks.end()) return;
    arena_free.emplace(it->second, off);
    live_blocks.erase(it);
}

// ---------------------------------------------------------------------------
// Descriptor handles
//
// A cuDNN descriptor is an opaque handle the guest only ever hands back, so it
// was a host pointer.  That is the same mistake device memory made: a host
// address written into the guest's memory cannot survive a change of host, and
// a saved session is exactly that.  So handles are small integers now, and the
// host keeps the table.

std::map<uint64_t, void*> descriptors;
uint64_t next_handle = 1;

uint64_t remember_descriptor(void* d) {
    if (!d) return 0;
    descriptors[next_handle] = d;
    return next_handle++;
}

void* descriptor(uint64_t handle) {
    auto it = descriptors.find(handle);
    return it == descriptors.end() ? nullptr : it->second;
}

void forget_descriptor(uint64_t handle) { descriptors.erase(handle); }

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
// (264) - so 288 covers everything, and the handler reads only what its
// signature says.  Copying more than the guest allocated is harmless *unless*
// it runs off the end of a mapping, so this stops at the first page it cannot
// read.
//
// Page at a time, and asking first rather than catching a fault.
//
// Halving on a fault looks equivalent to this and is not: an argument sitting
// eight bytes before the end of the last mapped page would come back as a
// *four* byte read, and a pointer truncated to its low half is a wrong answer
// rather than a short one.  Stopping at a page boundary can only truncate an
// argument that straddles into unmapped memory, which is one the guest could
// not have written either.
//
// And asking beats catching.  This runs for every argument of every kernel -
// nearly two thousand crossings in a run - and a throw is not a cheap way to
// answer "is that page there", least of all in a WebAssembly build where
// exceptions go out through JavaScript.  Memory::is_mapped answers it directly.
uint64_t guest_read_tolerant(x86emu::Emulator& e, uint64_t addr, uint8_t* dst,
                             uint64_t want) {
    constexpr uint64_t kPage = x86emu::Memory::kPageSize;
    uint64_t done = 0;
    while (done < want) {
        uint64_t at = addr + done;
        if (!e.mem.is_mapped(at)) break;
        uint64_t n = kPage - (at & (kPage - 1));
        if (n > want - done) n = want - done;
        e.mem.read(at, dst + done, n);
        done += n;
    }
    return done;
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

// Both sides are guest addresses now, so this could go through e.mem.read and
// e.mem.write throughout and be correct.  It does not, because device memory is
// contiguous and a memcpy over it is one call where the page-walking path is
// one per 4 KiB - and cudaMemcpy is how a hundred megabytes of weights arrive.
int64_t do_memcpy(x86emu::Emulator& e, uint64_t dst, uint64_t src, uint64_t n) {
    if (!n) return 0;
    uint8_t* d = device_host(dst);
    const uint8_t* s = device_host(src);
    if (d && s) {
        std::memcpy(d, s, n);
    } else if (d) {
        e.mem.read(src, d, n);       // guest -> device
    } else if (s) {
        e.mem.write(dst, s, n);      // device -> guest
    } else {
        std::vector<uint8_t> tmp(n);
        e.mem.read(src, tmp.data(), n);
        e.mem.write(dst, tmp.data(), n);
    }
    return 0;
}

// One device address in the copied argument block, converted in place.  A word
// that is not one is left alone, which covers a null and an argument the guest
// never filled in.
void translate(uint8_t* at, uint64_t avail) {
    if (avail < 8) return;
    uint64_t word;
    std::memcpy(&word, at, sizeof word);
    if (!is_device(word)) return;
    uint8_t* host = device_host(word);
    std::memcpy(at, &host, sizeof host);
    if (sizeof host < 8) std::memset(at + sizeof host, 0, 8 - sizeof host);
}

int64_t do_launch(x86emu::Emulator& e, uint64_t name_addr, uint64_t args_addr) {
    double t0 = vvstub_timing ? vvstub_now() : 0;
    struct Account {
        double t;
        ~Account() { if (vvstub_timing) vvstub_account(VVSTUB_T_KERNEL, t); }
    } account{t0};

    std::string name = e.mem.read_cstring(name_addr);
    unsigned ptrs = 0, arrays = 0;
    int nargs = vvstub_kernel_layout(name.c_str(), &ptrs, &arrays);
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
        uint64_t got = p ? guest_read_tolerant(e, p, slot, kMaxArg) : 0;

        // An argument that is a device pointer is a *guest* address, and the
        // kernels dereference what they are given - so it is translated here.
        //
        // Only where the signature says there is one.  Searching the block for
        // words that look like addresses is a different question with a
        // different answer: arguments are packed, an eight-byte read starting
        // at a four-byte element count takes the upper half of whatever follows
        // it, and when that is a device pointer the pair reads as an address
        // 0x3000'0000'0000 plus the count - which is inside the arena.  It
        // converted the count into a pointer, and 13568 elements became a loop
        // of 1.5 billion.
        if (ptrs >> i & 1u) translate(slot, got);
        if (arrays >> i & 1u) {
            // TArray<const void*, 32>: a count, then the addresses at offset 8.
            int32_t count = 0;
            if (got >= 4) std::memcpy(&count, slot, sizeof count);
            if (count < 0) count = 0;
            if (count > 32) count = 32;
            for (int k = 0; k < count; k++) {
                uint64_t off = 8 + 8 * static_cast<uint64_t>(k);
                if (off + 8 > got) break;
                translate(slot + off, got - off);
            }
        }
        host_args[static_cast<size_t>(i)] = slot;
    }
    return vvstub_run_kernel(name.c_str(), host_args.data()) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Capturing the state, to find out what a resume would cost
//
// Building the sessions takes minutes and synthesising takes seconds, so the
// obvious thing to want is to do the first once and start from there.  Whether
// that is *practical* turns on a number nobody has: how big is the state, and
// how much of it compresses.
//
// This writes it; restoring is the other half and is not written.  What goes in
// is every guest page with memory behind it - a reserved but untouched page
// reads as zero and can be left out - plus the device arena, because the guest
// is by now full of pointers into it.
//
// The file contains the decrypted voice model, in guest memory and again in the
// arena.  On your own disk that is your own process's state; it is not
// something to publish, and this writes it only when asked by name.

// Does this page still hold what the file it was mapped from holds?
//
// Files are kept open across the walk: a 98 MB dictionary is 24000 pages, and
// opening it 24000 times would take longer than writing the snapshot.
bool page_matches_file(const x86emu::Memory::Region& r, uint64_t addr,
                       const uint8_t* data) {
    constexpr uint64_t kPage = x86emu::Memory::kPageSize;
    static std::map<std::string, std::FILE*> open_files;
    auto it = open_files.find(r.file);
    if (it == open_files.end()) {
        std::string host = x86emu::FileTable::host_path(r.file);
        it = open_files.emplace(r.file, std::fopen(host.c_str(), "rb")).first;
    }
    std::FILE* f = it->second;
    if (!f) return false;
    if (std::fseek(f, static_cast<long>(r.file_offset + (addr - r.base)), SEEK_SET) != 0)
        return false;
    uint8_t buf[kPage];
    if (std::fread(buf, 1, kPage, f) != kPage) return false;
    return std::memcmp(buf, data, kPage) == 0;
}

uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

int64_t write_snapshot(x86emu::Emulator& e, const std::string& path) {
    constexpr uint64_t kPage = x86emu::Memory::kPageSize;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "[snap] cannot write %s\n", path.c_str());
        return -1;
    }

    std::vector<uint64_t> pages = e.mem.live_pages();
    uint64_t zero = 0, written = 0;
    std::vector<uint64_t> hashes;
    hashes.reserve(pages.size());
    // Where the pages are, by mapping.  A page that is a library's own image
    // could be re-read from the file on resume rather than carried, and this is
    // what says whether that is worth the machinery - guessing at it is how you
    // end up building the wrong half.
    std::map<std::string, uint64_t> by_region;
    uint64_t from_file = 0;
    for (uint64_t index : pages) {
        const uint8_t* data = e.mem.page_data(index);
        if (!data) continue;
        bool all_zero = true;
        for (uint64_t i = 0; i < kPage && all_zero; i++)
            if (data[i]) all_zero = false;
        if (all_zero) {
            zero++;
            continue;
        }
        uint64_t addr = index * kPage;
        const x86emu::Memory::Region* home = nullptr;
        for (const auto& r : e.mem.regions())
            if (addr >= r.base && addr < r.base + r.size) home = &r;
        by_region[home ? home->name : std::string("(anonymous)")]++;

        // A page that still matches the file it was mapped from does not have
        // to be carried: a resume can read it back.  Comparing is exact where a
        // dirty bit would need the emulator to keep one, and the guest here has
        // a 98 MB dictionary mapped that it never writes to.
        if (home && !home->file.empty() && page_matches_file(*home, addr, data)) {
            from_file++;
            continue;
        }

        hashes.push_back(fnv1a(data, kPage));
        std::fwrite(&index, sizeof index, 1, f);
        std::fwrite(data, 1, kPage, f);
        written++;
    }

    std::vector<std::pair<uint64_t, std::string>> ranked;
    for (const auto& [name, n] : by_region) ranked.push_back({n, name});
    std::sort(ranked.rbegin(), ranked.rend());
    for (size_t i = 0; i < ranked.size() && i < 12; i++)
        std::fprintf(stderr, "[snap]   %8.1f MB  %s\n",
                     ranked[i].first * kPage / 1048576.0, ranked[i].second.c_str());

    // Nothing here about the device arena, deliberately.  It used to be written
    // separately because it was host memory; it is guest memory now, so it came
    // through the page walk above with everything else - and a resume will get
    // it back the same way, at the same guest addresses the guest's own
    // pointers refer to.
    long total = std::ftell(f);
    std::fclose(f);

    std::sort(hashes.begin(), hashes.end());
    uint64_t unique = static_cast<uint64_t>(
        std::unique(hashes.begin(), hashes.end()) - hashes.begin());

    std::fprintf(stderr,
                 "[snap] %llu live pages: %llu written, %llu all-zero, %llu still "
                 "matching their file, %llu of the written distinct\n",
                 (unsigned long long)pages.size(), (unsigned long long)written,
                 (unsigned long long)zero, (unsigned long long)from_file,
                 (unsigned long long)unique);
    std::fprintf(stderr, "[snap] left behind: %.1f MB of file-backed pages\n",
                 from_file * kPage / 1048576.0);
    std::fprintf(stderr, "[snap] %.1f MB raw (device memory included, since it is\n"
                         "[snap] guest memory now)\n",
                 total / 1048576.0);
    return total;
}

// The whole boundary, timed as one bucket.  What a run costs beyond this is
// the guest, still interpreted - and on this workload that is now the larger
// half, which was not true before.
constexpr int kBoundaryBucket = VVSTUB_T_COUNT;

struct Report {
    ~Report() {
        if (!vvstub_timing) return;
        static const char* names[] = {"kernels", "cuDNN", "cuBLAS", "boundary"};
        for (int i = 0; i <= VVSTUB_T_COUNT; i++)
            std::fprintf(stderr, "[host] %-9s %8.3f s  %lld calls\n", names[i],
                         vvhost_seconds[i], vvhost_calls[i]);
        std::fprintf(stderr,
                     "[host] the rest of the run is the guest, interpreted\n");
    }
} report_at_exit;

}  // namespace

// ---------------------------------------------------------------------------

int64_t vv_host_call(x86emu::Emulator& e, uint64_t id, uint64_t argp) {
    // The kernels read pointers out of device memory in four places - a concat's
    // list of inputs, a split's list of outputs - and those are guest addresses
    // too.  Told once, on the first call.
    static bool told = [] {
        vvstub_set_device_host([](unsigned long long p) -> void* {
            return device_host(p);
        });
        return true;
    }();
    (void)told;

    double t_entry = vvstub_timing ? vvstub_now() : 0;
    struct Account {
        double t;
        ~Account() { if (vvstub_timing) vvstub_account(kBoundaryBucket, t); }
    } account{t_entry};

    uint64_t a[VVHOST_SLOTS];
    for (int i = 0; i < VVHOST_SLOTS; i++) a[i] = e.mem.read64(argp + 8u * i);


    switch (id) {
        case VVH_ALIVE:
            return 1;

        case VVH_SNAPSHOT:
            return write_snapshot(e, e.mem.read_cstring(a[0]));

        case VVH_MALLOC:
            return static_cast<int64_t>(device_alloc(e, a[0]));
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
                std::memset(device_host(a[0]), static_cast<int>(a[1]), a[2]);
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
            return static_cast<int64_t>(remember_descriptor(d));
        }
        case VVH_CUDNN_DESTROY_TENSOR: {
            int rc = cudnnDestroyTensorDescriptor(descriptor(a[0]));
            forget_descriptor(a[0]);
            return rc;
        }
        case VVH_CUDNN_CREATE_FILTER: {
            void* d = nullptr;
            cudnnCreateFilterDescriptor(&d);
            return static_cast<int64_t>(remember_descriptor(d));
        }
        case VVH_CUDNN_DESTROY_FILTER: {
            int rc = cudnnDestroyFilterDescriptor(descriptor(a[0]));
            forget_descriptor(a[0]);
            return rc;
        }
        case VVH_CUDNN_CREATE_CONV: {
            void* d = nullptr;
            cudnnCreateConvolutionDescriptor(&d);
            return static_cast<int64_t>(remember_descriptor(d));
        }
        case VVH_CUDNN_DESTROY_CONV: {
            int rc = cudnnDestroyConvolutionDescriptor(descriptor(a[0]));
            forget_descriptor(a[0]);
            return rc;
        }

        case VVH_CUDNN_SET_TENSOR_ND: {
            int nb = static_cast<int>(a[2]);
            std::vector<int> dim = guest_ints(e, a[3], nb);
            std::vector<int> stride = a[4] ? guest_ints(e, a[4], nb) : std::vector<int>();
            return cudnnSetTensorNdDescriptor(descriptor(a[0]), static_cast<int>(a[1]), nb,
                                              dim.data(),
                                              stride.empty() ? nullptr : stride.data());
        }
        case VVH_CUDNN_SET_TENSOR_4D:
            return cudnnSetTensor4dDescriptor(
                descriptor(a[0]), static_cast<int>(a[1]), static_cast<int>(a[2]),
                static_cast<int>(a[3]), static_cast<int>(a[4]), static_cast<int>(a[5]),
                static_cast<int>(a[6]));
        case VVH_CUDNN_SET_FILTER_ND: {
            int nb = static_cast<int>(a[3]);
            std::vector<int> dim = guest_ints(e, a[4], nb);
            return cudnnSetFilterNdDescriptor(descriptor(a[0]), static_cast<int>(a[1]),
                                              static_cast<int>(a[2]), nb, dim.data());
        }
        case VVH_CUDNN_SET_CONV_ND: {
            int nb = static_cast<int>(a[1]);
            std::vector<int> pad = guest_ints(e, a[2], nb);
            std::vector<int> stride = guest_ints(e, a[3], nb);
            std::vector<int> dil = guest_ints(e, a[4], nb);
            return cudnnSetConvolutionNdDescriptor(descriptor(a[0]), nb, pad.data(),
                                                   stride.data(), dil.data(),
                                                   static_cast<int>(a[5]),
                                                   static_cast<int>(a[6]));
        }
        case VVH_CUDNN_SET_CONV_GROUPS:
            return cudnnSetConvolutionGroupCount(descriptor(a[0]), static_cast<int>(a[1]));

        case VVH_CUDNN_CONV_FORWARD: {
            float alpha = guest_float(e, a[0], 1.0f);
            float beta = guest_float(e, a[7], 0.0f);
            return cudnnConvolutionForward(
                nullptr, &alpha, descriptor(a[1]), device_host(a[2]), descriptor(a[3]),
                device_host(a[4]), descriptor(a[5]), static_cast<int>(a[6]), nullptr, 0,
                &beta, descriptor(a[8]), device_host(a[9]));
        }
        case VVH_CUDNN_CONV_BACKWARD_DATA: {
            float alpha = guest_float(e, a[0], 1.0f);
            float beta = guest_float(e, a[7], 0.0f);
            return cudnnConvolutionBackwardData(
                nullptr, &alpha, descriptor(a[1]), device_host(a[2]), descriptor(a[3]),
                device_host(a[4]), descriptor(a[5]), static_cast<int>(a[6]), nullptr, 0,
                &beta, descriptor(a[8]), device_host(a[9]));
        }
        case VVH_CUDNN_ADD_TENSOR: {
            float alpha = guest_float(e, a[0], 1.0f);
            float beta = guest_float(e, a[3], 0.0f);
            return cudnnAddTensor(nullptr, &alpha, descriptor(a[1]), device_host(a[2]),
                                  &beta, descriptor(a[4]), device_host(a[5]));
        }

        // ---- cuBLAS ---------------------------------------------------------
        case VVH_CUBLAS_SGEMM: {
            float alpha = guest_float(e, a[5], 1.0f);
            float beta = guest_float(e, a[10], 0.0f);
            return cublasSgemm_v2(
                nullptr, static_cast<int>(a[0]), static_cast<int>(a[1]),
                static_cast<int>(a[2]), static_cast<int>(a[3]), static_cast<int>(a[4]),
                &alpha, device_as<const float>(a[6]),
                static_cast<int>(a[7]), device_as<const float>(a[8]),
                static_cast<int>(a[9]), &beta,
                device_as<float>(a[11]), static_cast<int>(a[12]));
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
                &alpha, device_as<const float>(a[6]),
                static_cast<int>(a[7]), extra[0],
                device_as<const float>(a[8]), static_cast<int>(a[9]),
                extra[1], &beta, device_as<float>(a[11]),
                static_cast<int>(a[12]), extra[2], static_cast<int>(extra[3]));
        }
        case VVH_CUBLAS_SGEAM: {
            float alpha = guest_float(e, a[4], 1.0f);
            float beta = guest_float(e, a[7], 0.0f);
            return cublasSgeam(nullptr, static_cast<int>(a[0]), static_cast<int>(a[1]),
                               static_cast<int>(a[2]), static_cast<int>(a[3]), &alpha,
                               device_as<const float>(a[5]),
                               static_cast<int>(a[6]), &beta,
                               device_as<const float>(a[8]),
                               static_cast<int>(a[9]),
                               device_as<float>(a[10]),
                               static_cast<int>(a[11]));
        }

        default:
            std::fprintf(stderr, "[cuda] host call %llu is not implemented\n",
                         static_cast<unsigned long long>(id));
            return -1;
    }
}
