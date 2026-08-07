// The one call the guest makes to reach the host, and what the ids mean.
//
// Shared by both sides: the guest stand-ins (guest/cudastub, built for x86-64
// Linux) and the host dispatcher (src/cudahost.cpp, compiled into the emulator).
// Keeping the numbering in one file is the only thing stopping the two halves
// from drifting apart silently.
//
// ---------------------------------------------------------------------------
// Why this works at all
//
// A CUDA device pointer is not host memory.  Nothing in ONNX Runtime ever
// dereferences one - it cannot, that is what "device" means - so device memory
// does not have to live in the guest's address space either.  cudaMalloc here
// returns a real host pointer and the guest passes it back and forth as an
// opaque number.  Tensors therefore need no marshalling at all, which is what
// makes this cheap: 377 kernel launches move no bytes.
//
// The exceptions are the things the guest genuinely reads:
//
//   * pinned host memory (cudaMallocHost) - that stays guest memory, allocated
//     by the guest's own malloc, and never crosses.
//   * out-parameters, where the host writes a handle into guest memory.
//   * cudaMemcpy between the two, which is the only place bytes are copied.
//
// ---------------------------------------------------------------------------
// The calling convention
//
// One reserved syscall, an id, and a pointer to sixteen 64-bit slots.  Every
// entry point in these libraries takes integers and pointers - the floats
// travel by pointer, which is cuBLAS's own convention - so sixteen slots
// covers all of them with room to spare (cudnnConvolutionForward, the widest,
// has thirteen).
#ifndef VVHOSTCALL_H
#define VVHOSTCALL_H

#define VVHOST_SYSCALL 0x7654321
#define VVHOST_SLOTS 16

// The id space is grouped so that a stray number is obvious in a trace.
enum {
    // Memory and launching: the parts that are not simply "do this arithmetic".
    VVH_MALLOC = 1,        // (size) -> device pointer, or 0
    VVH_FREE,              // (ptr)
    VVH_MEMCPY,            // (dst, src, bytes, kind)
    VVH_MEMSET,            // (ptr, value, bytes)
    VVH_MEMCPY2D,          // (dst, dpitch, src, spitch, width, height, kind)
    VVH_LAUNCH,            // (mangled name in guest memory, guest args array)
    VVH_ALIVE = 15,        // () -> 1; how the guest finds out the host is there
    // (path in guest memory) -> bytes written, or -1.  Writes the guest's whole
    // state through the emulator, and the shim's own bookkeeping - the arena's
    // free list and the descriptor table - beside it as PATH.shim.  Resuming is
    // vvcudaemu's --resume, which is not a host call: it happens before the
    // guest starts.
    // Taking one also reports where its weight is, by mapping, which is what
    // the measurement this started as was for.  Two ways of answering one
    // question is one too many.
    VVH_SNAPSHOT = 16,

    // cuDNN.  Descriptors are host objects; the guest only ever holds handles.
    VVH_CUDNN_CREATE_TENSOR = 100,
    VVH_CUDNN_DESTROY_TENSOR,
    VVH_CUDNN_SET_TENSOR_ND,
    VVH_CUDNN_SET_TENSOR_4D,
    VVH_CUDNN_CREATE_FILTER,
    VVH_CUDNN_DESTROY_FILTER,
    VVH_CUDNN_SET_FILTER_ND,
    VVH_CUDNN_CREATE_CONV,
    VVH_CUDNN_DESTROY_CONV,
    VVH_CUDNN_SET_CONV_ND,
    VVH_CUDNN_SET_CONV_GROUPS,
    VVH_CUDNN_CONV_FORWARD,
    VVH_CUDNN_CONV_BACKWARD_DATA,
    VVH_CUDNN_ADD_TENSOR,

    // cuBLAS.
    VVH_CUBLAS_SGEMM = 200,
    VVH_CUBLAS_SGEMM_BATCHED,
    VVH_CUBLAS_SGEAM,
};

// cudaMemcpyKind, which decides which side of the boundary each pointer is on.
enum {
    VVH_H2H = 0,  // guest  -> guest
    VVH_H2D = 1,  // guest  -> host
    VVH_D2H = 2,  // host   -> guest
    VVH_D2D = 3,  // host   -> host
    VVH_DEFAULT = 4,
};

#ifndef VVHOSTCALL_HOST_SIDE
// The one instruction that leaves the guest.  Linux syscall numbers are three
// digits, so the reserved one cannot collide; a real kernel answers ENOSYS and
// a guest not built for this can never reach it by accident.
static inline long vvhost(long id, const long* args) {
    long ret;
    __asm__ volatile("syscall"
                     : "=a"(ret)
                     : "a"((long)VVHOST_SYSCALL), "D"(id), "S"(args)
                     : "rcx", "r11", "memory");
    return ret;
}

// Packing an argument list.  Sixteen slots is more than the widest entry point
// needs, and zeroing the rest means a host handler reading one slot too many
// gets a zero rather than whatever the last call left there.
#define VVA(...)                          \
    long a[VVHOST_SLOTS] = {__VA_ARGS__}; \
    (void)a
#endif

#endif  // VVHOSTCALL_H
