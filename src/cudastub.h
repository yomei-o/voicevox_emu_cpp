// Shared by the generated CUDA stand-ins.
//
// A stub returns 0, which is `cudaSuccess` and `CUBLAS_STATUS_SUCCESS` and
// `CUDNN_STATUS_SUCCESS` alike - the three libraries agree that zero is fine.
// It declares no parameters, so a caller may pass whatever it likes: the
// System V ABI puts arguments in caller-saved registers and requires no
// clean-up, so ignoring them is safe.
//
// VVSTUB_TRACE=1 makes every stub announce itself, which is how you find the
// call that mattered after something goes wrong.
//
// The linkage matters.  libcudnn's stand-in is built as C++ because its real
// entry points use Eigen, and a C++ compiler would give every stub a mangled
// name - which is not the name the provider imports, and shows up as
// `undefined symbol: _Z11vvstub_notePKc` at load.
#ifndef CUDASTUB_H
#define CUDASTUB_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
#define VVSTUB_C extern "C"
extern "C" {
#else
#define VVSTUB_C
#endif

extern int vvstub_trace;
void vvstub_note(const char* name);

// VVSTUB_TIME=1: where the shim's own seconds go.  The CUDA provider hands
// *all* of its arithmetic across this boundary, so what these three buckets add
// up to is the arithmetic, and whatever the caller measured beyond it is ONNX
// Runtime's own plumbing.  That split is the thing to know before deciding
// whether moving the arithmetic out of an emulator is worth the work.
enum { VVSTUB_T_KERNEL, VVSTUB_T_CUDNN, VVSTUB_T_CUBLAS, VVSTUB_T_COUNT };
extern int vvstub_timing;
double vvstub_now(void);
void vvstub_account(int bucket, double started);

// Implemented in cudakernels.c: does the kernel natively when it knows how,
// and answers 0 when it does not.
int vvstub_run_kernel(const char* name, void** args);
// How many arguments it takes, or 0 for one this build does not know.  A
// caller that has to marshal them across a boundary needs the count, and
// `cudaLaunchKernel` does not carry one.
int vvstub_kernel_nargs(const char* name);
// The same, and also which arguments carry device addresses: a bit per index in
// `ptrs` for an address on its own, and in `arrays` for a TArray of them
// arriving by value.  A caller that has to convert those addresses must not
// guess which words are addresses - an element count beside the upper half of a
// pointer reads as one.
int vvstub_kernel_layout(const char* name, unsigned* ptrs, unsigned* arrays);
// The three kinds of cuDNN descriptor this build has, for a caller that has to
// write the table down and build it again later.  Implemented in cudnn_real.cpp,
// where the structs are: all three are fixed-width integers and nothing else, so
// their bytes carry between a host with eight-byte pointers and one with four.
#define VVSTUB_DESC_TENSOR 1
#define VVSTUB_DESC_FILTER 2
#define VVSTUB_DESC_CONV   3
int vvstub_descriptor_size(int kind);
void* vvstub_descriptor_new(int kind);

// How to dereference a device pointer, when it is not already a host one.
// Unset natively; set by the emulator's host shim, where device memory lives at
// a guest address.
void vvstub_set_device_host(void* (*fn)(unsigned long long));

#ifdef __cplusplus
}
#endif

#define VVSTUB(name)                          \
    VVSTUB_C int name();                      \
    VVSTUB_C int name() {                     \
        if (vvstub_trace) vvstub_note(#name); \
        return 0;                             \
    }

#endif  // CUDASTUB_H
