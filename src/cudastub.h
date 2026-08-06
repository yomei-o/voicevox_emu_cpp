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
#ifndef CUDASTUB_H
#define CUDASTUB_H

#include <stdio.h>
#include <stdlib.h>

extern int vvstub_trace;
void vvstub_note(const char* name);

// Implemented in cudakernels.c: does the kernel natively when it knows how,
// and answers 0 when it does not.
int vvstub_run_kernel(const char* name, void** args);

#define VVSTUB(name)                          \
    int name();                               \
    int name() {                              \
        if (vvstub_trace) vvstub_note(#name); \
        return 0;                             \
    }

#endif  // CUDASTUB_H
