// The cuDNN entry points that have to compute, done natively with Eigen.
//
// Of the sixty-one cuDNN functions ONNX Runtime's CUDA provider imports, a real
// model calls fifteen, and only three of those do arithmetic: the convolution,
// its transposed form, and the bias add.  The other twelve create and fill
// descriptors, which is bookkeeping - numerous (291 tensor descriptors for one
// utterance) and trivial.
//
// The convolution is done the way every CPU implementation does it: im2col into
// a matrix, one GEMM, and let Eigen have the inner loop.  That is not how a GPU
// would do it, but it is what "run the CUDA build with no GPU" means.
//
// Everything here is float, NCHW, which is what these models use.  Anything
// else returns CUDNN_STATUS_NOT_SUPPORTED rather than quietly computing the
// wrong thing - a silently wrong answer is the failure mode this whole project
// spent a day learning to avoid.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "Eigen_Core.h"
#include "cudastub.h"

// VVSTUB_TIME=1 attributes these seconds to the cuDNN bucket; see cudastub.h
// for what the split is for.
namespace {

struct Timed {
    double t0;
    Timed() : t0(vvstub_timing ? vvstub_now() : 0) {}
    ~Timed() { if (vvstub_timing) vvstub_account(VVSTUB_T_CUDNN, t0); }
};


constexpr int kOk = 0;               // CUDNN_STATUS_SUCCESS
constexpr int kNotSupported = 9;     // CUDNN_STATUS_NOT_SUPPORTED
constexpr int kDataTypeFloat = 0;    // CUDNN_DATA_FLOAT

// cuDNN's descriptors are opaque handles.  Ours are objects we allocate, which
// is all a handle has to be.

struct TensorDesc {
    int type = kDataTypeFloat;
    int nb = 0;
    int dim[8] = {0};
    int stride[8] = {0};
    long long count() const {
        long long n = 1;
        for (int i = 0; i < nb; i++) n *= dim[i];
        return n;
    }
};

struct FilterDesc {
    int type = kDataTypeFloat;
    int format = 0;
    int nb = 0;
    int dim[8] = {0};  // [out_channels, in_channels/groups, kernel...]
};

struct ConvDesc {
    int nb = 0;               // number of spatial dimensions
    int pad[4] = {0};
    int stride[4] = {1, 1, 1, 1};
    int dilation[4] = {1, 1, 1, 1};
    int mode = 1;             // 1 = CUDNN_CROSS_CORRELATION, which ONNX uses
    int groups = 1;
};

using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixMap = Eigen::Map<Matrix>;
using ConstMatrixMap = Eigen::Map<const Matrix>;

// A spatial layout reduced to the one shape everything else is a case of:
// batch, channels, and a single spatial extent.  A 1-D convolution is this
// directly; a 2-D one with height 1 - which is how these models are stored -
// is the same thing.
struct Shape1D {
    int n = 1, c = 1, len = 1;
    // Which spatial slot the extent came from, counting from the first spatial
    // dimension.  The convolution descriptor's pad/stride/dilation arrays are in
    // that same order, so this is the index to read them at.
    int sp = 0;
    bool ok = false;
};

Shape1D as_1d(const TensorDesc& t) {
    Shape1D s;
    if (t.nb == 3) {  // NCW
        s.n = t.dim[0];
        s.c = t.dim[1];
        s.len = t.dim[2];
        s.sp = 0;
        s.ok = true;
    } else if (t.nb == 4) {  // NCHW, with one of the spatial dims degenerate
        s.n = t.dim[0];
        s.c = t.dim[1];
        // ONNX Runtime stores a 1-D convolution as [N, C, W, 1] - the extent is
        // the *height* slot and the width is the degenerate one.  Reading the
        // last slot instead gets pad = 0 where the model asked for 2, and every
        // convolution then shifts its output by that much.
        if (t.dim[3] == 1) {
            s.len = t.dim[2];
            s.sp = 0;
            s.ok = true;
        } else if (t.dim[2] == 1) {
            s.len = t.dim[3];
            s.sp = 1;
            s.ok = true;
        }
    }
    return s;
}

// im2col for one batch item of a 1-D convolution: rows are output positions,
// columns are (input channel, tap).  The matrix that comes out is exactly what
// a GEMM against the flattened filter needs.
void im2col(const float* x, int c_in, int in_len, int k, int pad, int stride,
            int dilation, int out_len, Matrix& col) {
    col.resize(out_len, (long)c_in * k);
    for (int o = 0; o < out_len; o++) {
        for (int ci = 0; ci < c_in; ci++) {
            for (int t = 0; t < k; t++) {
                int i = o * stride - pad + t * dilation;
                col(o, (long)ci * k + t) =
                    (i < 0 || i >= in_len) ? 0.0f : x[(long)ci * in_len + i];
            }
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// The C interface.  Signatures follow cuDNN 8; only the parameters that matter
// are named.

extern "C" {

int cudnnCreateTensorDescriptor(void** d) { *d = new TensorDesc(); return kOk; }
int cudnnDestroyTensorDescriptor(void* d) { delete (TensorDesc*)d; return kOk; }

int cudnnSetTensorNdDescriptor(void* d, int type, int nb, const int* dim,
                               const int* stride) {
    auto* t = (TensorDesc*)d;
    t->type = type;
    t->nb = nb > 8 ? 8 : nb;
    for (int i = 0; i < t->nb; i++) {
        t->dim[i] = dim[i];
        t->stride[i] = stride ? stride[i] : 0;
    }
    return kOk;
}

int cudnnSetTensor4dDescriptor(void* d, int format, int type, int n, int c, int h,
                               int w) {
    auto* t = (TensorDesc*)d;
    (void)format;
    t->type = type;
    t->nb = 4;
    t->dim[0] = n; t->dim[1] = c; t->dim[2] = h; t->dim[3] = w;
    t->stride[3] = 1; t->stride[2] = w; t->stride[1] = h * w; t->stride[0] = c * h * w;
    return kOk;
}

int cudnnCreateFilterDescriptor(void** d) { *d = new FilterDesc(); return kOk; }
int cudnnDestroyFilterDescriptor(void* d) { delete (FilterDesc*)d; return kOk; }

int cudnnSetFilterNdDescriptor(void* d, int type, int format, int nb,
                               const int* dim) {
    auto* f = (FilterDesc*)d;
    f->type = type;
    f->format = format;
    f->nb = nb > 8 ? 8 : nb;
    for (int i = 0; i < f->nb; i++) f->dim[i] = dim[i];
    return kOk;
}

int cudnnCreateConvolutionDescriptor(void** d) { *d = new ConvDesc(); return kOk; }
int cudnnDestroyConvolutionDescriptor(void* d) { delete (ConvDesc*)d; return kOk; }

int cudnnSetConvolutionNdDescriptor(void* d, int nb, const int* pad,
                                    const int* stride, const int* dilation, int mode,
                                    int computeType) {
    auto* c = (ConvDesc*)d;
    (void)computeType;
    c->nb = nb > 4 ? 4 : nb;
    for (int i = 0; i < c->nb; i++) {
        c->pad[i] = pad[i];
        c->stride[i] = stride[i];
        c->dilation[i] = dilation[i];
    }
    c->mode = mode;
    return kOk;
}

int cudnnSetConvolutionGroupCount(void* d, int groups) {
    ((ConvDesc*)d)->groups = groups;
    return kOk;
}

// No workspace is needed: im2col allocates its own.
int cudnnGetConvolutionForwardWorkspaceSize(void*, void*, void*, void*, void*, int,
                                            size_t* size) {
    if (size) *size = 0;
    return kOk;
}
int cudnnGetConvolutionBackwardDataWorkspaceSize(void*, void*, void*, void*, void*,
                                                 int, size_t* size) {
    if (size) *size = 0;
    return kOk;
}

// The algorithm search.  There is one algorithm here, so the answer is a single
// zeroed result: algo 0, status success, no time and no memory.  ORT reads the
// first entry and asks for nothing else.
int cudnnFindConvolutionForwardAlgorithmEx(void*, void*, const void*, void*,
                                           const void*, void*, void*, void*,
                                           int requested, int* returned,
                                           void* results, void*, size_t) {
    if (returned) *returned = 1;
    if (results && requested > 0) std::memset(results, 0, 64);
    return kOk;
}

int cudnnFindConvolutionBackwardDataAlgorithmEx(void*, void*, const void*, void*,
                                                const void*, void*, void*, void*,
                                                int requested, int* returned,
                                                void* results, void*, size_t) {
    if (returned) *returned = 1;
    if (results && requested > 0) std::memset(results, 0, 64);
    return kOk;
}

// C = alpha * A + beta * C, where A may be shorter than C and repeats.  This is
// how a bias reaches every position of every batch item.
int cudnnAddTensor(void*, const void* alpha, void* aDesc, const void* A,
                   const void* beta, void* cDesc, void* C) {
    Timed timed;
    const auto* a = (const TensorDesc*)aDesc;
    const auto* c = (const TensorDesc*)cDesc;
    if (a->type != kDataTypeFloat || c->type != kDataTypeFloat) return kNotSupported;

    float al = alpha ? *(const float*)alpha : 1.0f;
    float be = beta ? *(const float*)beta : 0.0f;
    const float* src = (const float*)A;
    float* dst = (float*)C;

    Shape1D as = as_1d(*a), cs = as_1d(*c);
    if (!as.ok || !cs.ok) return kNotSupported;
    // The bias case: one value per channel, applied across batch and length.
    if (as.c == cs.c && as.n == 1 && as.len == 1) {
        for (int n = 0; n < cs.n; n++)
            for (int ch = 0; ch < cs.c; ch++) {
                float add = al * src[ch];
                float* row = dst + ((long)n * cs.c + ch) * cs.len;
                for (int i = 0; i < cs.len; i++) row[i] = be * row[i] + add;
            }
        return kOk;
    }
    // The plain case: same shape, element for element.
    if (as.n == cs.n && as.c == cs.c && as.len == cs.len) {
        long total = (long)cs.n * cs.c * cs.len;
        for (long i = 0; i < total; i++) dst[i] = be * dst[i] + al * src[i];
        return kOk;
    }
    return kNotSupported;
}

// y = alpha * conv(x, w) + beta * y
int cudnnConvolutionForward(void*, const void* alpha, void* xDesc, const void* x,
                            void* wDesc, const void* w, void* convDesc, int algo,
                            void*, size_t, const void* beta, void* yDesc, void* y) {
    Timed timed;
    (void)algo;
    const auto* xd = (const TensorDesc*)xDesc;
    const auto* wd = (const FilterDesc*)wDesc;
    const auto* cd = (const ConvDesc*)convDesc;
    const auto* yd = (const TensorDesc*)yDesc;
    if (xd->type != kDataTypeFloat || wd->type != kDataTypeFloat) return kNotSupported;

    Shape1D xs = as_1d(*xd), ys = as_1d(*yd);
    if (!xs.ok || !ys.ok) return kNotSupported;

    // The filter is [out_channels, in_channels/groups, taps], with the taps
    // spread over however many spatial dimensions the descriptor has.
    int out_c = wd->dim[0];
    int in_c_per_group = wd->dim[1];
    int k = 1;
    for (int i = 2; i < wd->nb; i++) k *= wd->dim[i];

    int groups = cd->groups > 0 ? cd->groups : 1;
    // Read pad/stride/dilation at the slot the extent came from, not at a fixed
    // one - see as_1d.
    int sp = xs.sp < cd->nb ? xs.sp : 0;
    int pad = cd->pad[sp], stride = cd->stride[sp], dilation = cd->dilation[sp];

    if (getenv("VVSTUB_TRACE")) {
        auto dims = [](const int* d, int nb, char* buf) {
            int o = 0;
            for (int i = 0; i < nb; i++) o += sprintf(buf + o, "%s%d", i ? "," : "", d[i]);
            if (!nb) buf[0] = 0;
            return buf;
        };
        char bx[64], bw[64], by[64], bp[64], bs[64], bd[64];
        fprintf(stderr,
                "[conv] x[%d]=%s w[%d]=%s y[%d]=%s conv nb=%d pad=%s stride=%s dil=%s "
                "groups=%d\n",
                xd->nb, dims(xd->dim, xd->nb, bx), wd->nb, dims(wd->dim, wd->nb, bw),
                yd->nb, dims(yd->dim, yd->nb, by), cd->nb, dims(cd->pad, cd->nb, bp),
                dims(cd->stride, cd->nb, bs), dims(cd->dilation, cd->nb, bd), groups);
    }

    float al = alpha ? *(const float*)alpha : 1.0f;
    float be = beta ? *(const float*)beta : 0.0f;
    const float* xp = (const float*)x;
    const float* wp = (const float*)w;
    float* yp = (float*)y;

    if (getenv("VVSTUB_STATS")) {
        double lo = 0, hi = 0, sum = 0;
        long total = (long)xs.n * xs.c * xs.len;
        for (long i = 0; i < total; i++) {
            float v = xp[i];
            if (i == 0 || v < lo) lo = v;
            if (i == 0 || v > hi) hi = v;
            sum += v;
        }
        fprintf(stderr, "[conv-in] %ldx%d min=%g max=%g mean=%g\n", (long)xs.c,
                xs.len, lo, hi, total ? sum / total : 0.0);
    }

    int out_c_per_group = out_c / groups;
    Matrix col;
    for (int n = 0; n < xs.n; n++) {
        for (int g = 0; g < groups; g++) {
            const float* xg = xp + ((long)n * xs.c + (long)g * in_c_per_group) * xs.len;
            float* yg = yp + ((long)n * ys.c + (long)g * out_c_per_group) * ys.len;
            ConstMatrixMap xm(xg, in_c_per_group, xs.len);
            MatrixMap ym(yg, out_c_per_group, ys.len);

            if (stride == 1) {
                // One GEMM per tap, straight out of the input.
                //
                // im2col materialises an out_len x (in_c * k) matrix, which for
                // this model's widest layer is 8000 x 1408 - 45 MB built and
                // read back for a single convolution.  The arithmetic is the
                // same either way; what differs is that this reads the input k
                // times instead of copying it k times, and 90 % of the shim's
                // time was here.
                if (be == 0.0f)
                    ym.setZero();
                else
                    ym *= be;
                for (int t = 0; t < k; t++) {
                    // Output positions whose tap t lands inside the input.
                    int shift = t * dilation - pad;
                    int o0 = shift < 0 ? -shift : 0;
                    int o1 = xs.len - shift;
                    if (o1 > ys.len) o1 = ys.len;
                    if (o1 <= o0) continue;
                    // W(oc, ic, t) has the taps innermost, so a fixed t is a
                    // strided view rather than a copy.
                    Eigen::Map<const Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
                        wt(wp + (long)g * out_c_per_group * in_c_per_group * k + t,
                           out_c_per_group, in_c_per_group,
                           Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                               (long)in_c_per_group * k, k));
                    ym.block(0, o0, out_c_per_group, o1 - o0).noalias() +=
                        al * (wt * xm.block(0, o0 + shift, in_c_per_group, o1 - o0));
                }
                continue;
            }

            // A strided convolution does not decompose that way; im2col still
            // does.  Nothing in these models takes this path.
            im2col(xg, in_c_per_group, xs.len, k, pad, stride, dilation, ys.len, col);

            // The filter block for this group, as [out_c_per_group, in*k].
            ConstMatrixMap wm(wp + (long)g * out_c_per_group * in_c_per_group * k,
                              out_c_per_group, (long)in_c_per_group * k);
            // [out_c_per_group, out_len] = W * colᵀ
            Matrix out = wm * col.transpose();

            for (int oc = 0; oc < out_c_per_group; oc++)
                for (int i = 0; i < ys.len; i++) {
                    float* dst = &yg[(long)oc * ys.len + i];
                    *dst = be * *dst + al * out(oc, i);
                }
        }
    }
    return kOk;
}

// The transposed convolution, which ONNX calls ConvTranspose and cuDNN reaches
// through the backward-data pass.  Each input position scatters its taps
// forward, which is the transpose of the gather the forward pass does.
int cudnnConvolutionBackwardData(void*, const void* alpha, void* wDesc,
                                 const void* w, void* dyDesc, const void* dy,
                                 void* convDesc, int algo, void*, size_t,
                                 const void* beta, void* dxDesc, void* dx) {
    Timed timed;
    (void)algo;
    const auto* wd = (const FilterDesc*)wDesc;
    const auto* dyd = (const TensorDesc*)dyDesc;
    const auto* cd = (const ConvDesc*)convDesc;
    const auto* dxd = (const TensorDesc*)dxDesc;
    if (wd->type != kDataTypeFloat) return kNotSupported;

    Shape1D ys = as_1d(*dyd), xs = as_1d(*dxd);
    if (!ys.ok || !xs.ok) return kNotSupported;

    int in_c = wd->dim[0];              // the "forward" input channels
    int out_c_per_group = wd->dim[1];   // and its output channels
    int k = 1;
    for (int i = 2; i < wd->nb; i++) k *= wd->dim[i];
    int groups = cd->groups > 0 ? cd->groups : 1;
    int sp = xs.sp < cd->nb ? xs.sp : 0;
    int pad = cd->pad[sp], stride = cd->stride[sp], dilation = cd->dilation[sp];

    float al = alpha ? *(const float*)alpha : 1.0f;
    float be = beta ? *(const float*)beta : 0.0f;
    const float* wp = (const float*)w;
    const float* yp = (const float*)dy;
    float* xp = (float*)dx;

    int in_c_per_group = in_c / groups;
    long total = (long)xs.n * xs.c * xs.len;
    if (be == 0.0f)
        std::memset(xp, 0, total * sizeof(float));
    else
        for (long i = 0; i < total; i++) xp[i] *= be;

    for (int n = 0; n < ys.n; n++)
        for (int g = 0; g < groups; g++)
            for (int ic = 0; ic < in_c_per_group; ic++) {
                const float* yc =
                    yp + ((long)n * ys.c + (long)g * in_c_per_group + ic) * ys.len;
                for (int oc = 0; oc < out_c_per_group; oc++) {
                    const float* wk =
                        wp + (((long)g * in_c_per_group + ic) * out_c_per_group + oc) * k;
                    float* xc =
                        xp + ((long)n * xs.c + (long)g * out_c_per_group + oc) * xs.len;
                    for (int o = 0; o < ys.len; o++) {
                        float v = al * yc[o];
                        if (v == 0.0f) continue;
                        for (int t = 0; t < k; t++) {
                            int i = o * stride - pad + t * dilation;
                            if (i >= 0 && i < xs.len) xc[i] += v * wk[t];
                        }
                    }
                }
            }
    return kOk;
}

}  // extern "C"
