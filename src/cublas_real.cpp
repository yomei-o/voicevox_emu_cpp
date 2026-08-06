// The cuBLAS entry points that have to compute, done with Eigen.
//
// Five of the twenty-three cuBLAS functions the CUDA provider imports are
// actually called by a model, and three of them do arithmetic: a general matrix
// multiply, its batched form, and the add-and-transpose that ONNX Runtime uses
// to move data between layouts.
//
// cuBLAS is column-major, which is the one thing to keep straight here.  Eigen
// is told so explicitly rather than by transposing anything: a column-major Map
// over the same bytes is free, and a transpose is not.
#include <cstddef>
#include <cstring>

#include "Eigen_Core.h"

namespace {

constexpr int kOk = 0;                   // CUBLAS_STATUS_SUCCESS
constexpr int kNotSupported = 15;        // CUBLAS_STATUS_NOT_SUPPORTED
constexpr int kOpN = 0;                  // CUBLAS_OP_N
constexpr int kOpT = 1;                  // CUBLAS_OP_T

using ColMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Stride = Eigen::OuterStride<Eigen::Dynamic>;
using ConstMap = Eigen::Map<const ColMajor, 0, Stride>;
using Map = Eigen::Map<ColMajor, 0, Stride>;

// A cuBLAS operand: `rows x cols` of the *stored* matrix, with `ld` between
// columns.  What op(A) is depends on the transpose flag, and the caller's m/n/k
// describe op(A), not A - which is the usual place to go wrong.
ConstMap view(const float* p, int rows, int cols, int ld) {
    return ConstMap(p, rows, cols, Stride(ld));
}

}  // namespace

extern "C" {

// The handle is opaque and nothing here needs state, so any non-null value will
// do; ONNX Runtime only ever passes it back.
int cublasCreate_v2(void** handle) {
    static int one;
    *handle = &one;
    return kOk;
}
int cublasDestroy_v2(void*) { return kOk; }
int cublasSetStream_v2(void*, void*) { return kOk; }
int cublasGetStream_v2(void*, void** s) { if (s) *s = nullptr; return kOk; }
int cublasSetMathMode(void*, int) { return kOk; }
int cublasGetMathMode(void*, int* m) { if (m) *m = 0; return kOk; }
int cublasSetPointerMode_v2(void*, int) { return kOk; }

// C = alpha * op(A) * op(B) + beta * C, all column-major.
int cublasSgemm_v2(void*, int transa, int transb, int m, int n, int k,
                   const float* alpha, const float* A, int lda, const float* B,
                   int ldb, const float* beta, float* C, int ldc) {
    float al = alpha ? *alpha : 1.0f;
    float be = beta ? *beta : 0.0f;

    // m, n and k describe op(A) and op(B), not A and B - which is the usual
    // place to go wrong.  op(A) is m x k, so A itself is k x m when transposed.
    ColMajor a = transa == kOpN ? ColMajor(view(A, m, k, lda))
                                : ColMajor(view(A, k, m, lda).transpose());
    ColMajor b = transb == kOpN ? ColMajor(view(B, k, n, ldb))
                                : ColMajor(view(B, n, k, ldb).transpose());
    Map c(C, m, n, Stride(ldc));

    if (be == 0.0f)
        c = al * (a * b);
    else
        c = al * (a * b) + be * c;
    return kOk;
}

// The same, for a run of matrices laid end to end at a fixed stride.
int cublasSgemmStridedBatched(void* h, int transa, int transb, int m, int n, int k,
                              const float* alpha, const float* A, int lda,
                              long long strideA, const float* B, int ldb,
                              long long strideB, const float* beta, float* C,
                              int ldc, long long strideC, int batch) {
    for (int i = 0; i < batch; i++) {
        int st = cublasSgemm_v2(h, transa, transb, m, n, k, alpha, A + i * strideA, lda,
                                B + i * strideB, ldb, beta, C + i * strideC, ldc);
        if (st != kOk) return st;
    }
    return kOk;
}

// C = alpha * op(A) + beta * op(B), which is how ONNX Runtime transposes.
int cublasSgeam(void*, int transa, int transb, int m, int n, const float* alpha,
                const float* A, int lda, const float* beta, const float* B, int ldb,
                float* C, int ldc) {
    float al = alpha ? *alpha : 1.0f;
    float be = beta ? *beta : 0.0f;
    Map c(C, m, n, Stride(ldc));

    // op(A) and op(B) are both m x n; A and B are stored the other way round
    // when their flag says so.
    if (transa == kOpN)
        c = al * view(A, m, n, lda);
    else
        c = al * view(A, n, m, lda).transpose();

    if (be != 0.0f) {
        if (transb == kOpN)
            c += be * view(B, m, n, ldb);
        else
            c += be * view(B, n, m, ldb).transpose();
    }
    return kOk;
}

}  // extern "C"
