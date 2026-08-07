// cuBLAS, guest side: trampolines to the host.
//
// Part of the emulated build of the stand-ins; see src/cudaguest.c for what the
// split is and src/vvhostcall.h for how the boundary works.  Nothing here
// computes - each entry point packs its arguments and hands them across.
#include "cudastub.h"
#include "vvhostcall.h"

#include <string.h>

// ---- cuBLAS ----------------------------------------------------------------
// alpha and beta are the only floats in any of these signatures, and cuBLAS
// passes them by pointer - into *guest* memory, since ORT keeps them on its own
// stack.  The host reads them from there.

int cublasCreate_v2(void** handle) {
    static int one;
    *handle = &one;
    return 0;
}
int cublasDestroy_v2(void* h) { (void)h; return 0; }
int cublasSetStream_v2(void* h, void* s) { (void)h; (void)s; return 0; }
int cublasGetStream_v2(void* h, void** s) { (void)h; if (s) *s = 0; return 0; }
int cublasSetMathMode(void* h, int m) { (void)h; (void)m; return 0; }
int cublasGetMathMode(void* h, int* m) { (void)h; if (m) *m = 0; return 0; }
int cublasSetPointerMode_v2(void* h, int m) { (void)h; (void)m; return 0; }

int cublasSgemm_v2(void* h, int transa, int transb, int m, int n, int k,
                   const float* alpha, const float* A, int lda, const float* B, int ldb,
                   const float* beta, float* C, int ldc) {
    (void)h;
    VVA(transa, transb, m, n, k, (long)alpha, (long)A, lda, (long)B, ldb, (long)beta,
        (long)C, ldc);
    return (int)vvhost(VVH_CUBLAS_SGEMM, a);
}
int cublasSgemmStridedBatched(void* h, int transa, int transb, int m, int n, int k,
                              const float* alpha, const float* A, int lda,
                              long long strideA, const float* B, int ldb,
                              long long strideB, const float* beta, float* C, int ldc,
                              long long strideC, int batch) {
    (void)h;
    // Seventeen arguments against sixteen slots, so the three strides and the
    // batch count go in one slot as a pointer to them - they are already
    // adjacent on the caller's stack in spirit, and this keeps VVHOST_SLOTS
    // from having to grow for one function.
    long long extra[4] = {strideA, strideB, strideC, batch};
    VVA(transa, transb, m, n, k, (long)alpha, (long)A, lda, (long)B, ldb, (long)beta,
        (long)C, ldc, (long)extra);
    return (int)vvhost(VVH_CUBLAS_SGEMM_BATCHED, a);
}
int cublasSgeam(void* h, int transa, int transb, int m, int n, const float* alpha,
                const float* A, int lda, const float* beta, const float* B, int ldb,
                float* C, int ldc) {
    (void)h;
    VVA(transa, transb, m, n, (long)alpha, (long)A, lda, (long)beta, (long)B, ldb,
        (long)C, ldc);
    return (int)vvhost(VVH_CUBLAS_SGEAM, a);
}
