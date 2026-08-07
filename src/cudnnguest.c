// cuDNN, guest side: trampolines to the host.
//
// Part of the emulated build of the stand-ins; see src/cudaguest.c for what the
// split is and src/vvhostcall.h for how the boundary works.  Nothing here
// computes - each entry point packs its arguments and hands them across.
#include "cudastub.h"
#include "vvhostcall.h"

#include <string.h>

// ---- cuDNN -----------------------------------------------------------------
// The descriptors are host objects.  The guest holds a handle it never looks
// inside, which is exactly what an opaque pointer is for.

#define VVH_CREATE(fn, id)                       \
    int fn(void** d) {                           \
        VVA(0);                                  \
        long r = vvhost(id, a);                  \
        *d = (void*)r;                           \
        return r ? 0 : 1;                        \
    }
#define VVH_DESTROY(fn, id)                      \
    int fn(void* d) {                            \
        VVA((long)d);                            \
        return (int)vvhost(id, a);               \
    }

VVH_CREATE(cudnnCreateTensorDescriptor, VVH_CUDNN_CREATE_TENSOR)
VVH_DESTROY(cudnnDestroyTensorDescriptor, VVH_CUDNN_DESTROY_TENSOR)
VVH_CREATE(cudnnCreateFilterDescriptor, VVH_CUDNN_CREATE_FILTER)
VVH_DESTROY(cudnnDestroyFilterDescriptor, VVH_CUDNN_DESTROY_FILTER)
VVH_CREATE(cudnnCreateConvolutionDescriptor, VVH_CUDNN_CREATE_CONV)
VVH_DESTROY(cudnnDestroyConvolutionDescriptor, VVH_CUDNN_DESTROY_CONV)

int cudnnSetTensorNdDescriptor(void* d, int type, int nb, const int* dim,
                               const int* stride) {
    VVA((long)d, type, nb, (long)dim, (long)stride);
    return (int)vvhost(VVH_CUDNN_SET_TENSOR_ND, a);
}
int cudnnSetTensor4dDescriptor(void* d, int format, int type, int n, int c, int h,
                               int w) {
    VVA((long)d, format, type, n, c, h, w);
    return (int)vvhost(VVH_CUDNN_SET_TENSOR_4D, a);
}
int cudnnSetFilterNdDescriptor(void* d, int type, int format, int nb, const int* dim) {
    VVA((long)d, type, format, nb, (long)dim);
    return (int)vvhost(VVH_CUDNN_SET_FILTER_ND, a);
}
int cudnnSetConvolutionNdDescriptor(void* d, int nb, const int* pad, const int* stride,
                                    const int* dilation, int mode, int computeType) {
    VVA((long)d, nb, (long)pad, (long)stride, (long)dilation, mode, computeType);
    return (int)vvhost(VVH_CUDNN_SET_CONV_ND, a);
}
int cudnnSetConvolutionGroupCount(void* d, int groups) {
    VVA((long)d, groups);
    return (int)vvhost(VVH_CUDNN_SET_CONV_GROUPS, a);
}

// No workspace is needed: the host implementation allocates its own.
int cudnnGetConvolutionForwardWorkspaceSize(void* h, void* x, void* w, void* c,
                                            void* y, int algo, size_t* size) {
    (void)h; (void)x; (void)w; (void)c; (void)y; (void)algo;
    if (size) *size = 0;
    return 0;
}
int cudnnGetConvolutionBackwardDataWorkspaceSize(void* h, void* w, void* dy, void* c,
                                                 void* dx, int algo, size_t* size) {
    (void)h; (void)w; (void)dy; (void)c; (void)dx; (void)algo;
    if (size) *size = 0;
    return 0;
}

// The algorithm search.  There is one algorithm, so the answer is a single
// zeroed result: algo 0, status success, no time and no memory.  ORT reads the
// first entry and asks for nothing else.  This can stay guest-side because the
// answer does not depend on anything the host knows.
int cudnnFindConvolutionForwardAlgorithmEx(void* h, void* xd, const void* x, void* wd,
                                           const void* w, void* cd, void* yd, void* y,
                                           int requested, int* returned, void* results,
                                           void* ws, size_t wsz) {
    (void)h; (void)xd; (void)x; (void)wd; (void)w; (void)cd; (void)yd; (void)y;
    (void)ws; (void)wsz;
    if (returned) *returned = 1;
    if (results && requested > 0) memset(results, 0, 64);
    return 0;
}
int cudnnFindConvolutionBackwardDataAlgorithmEx(void* h, void* wd, const void* w,
                                                void* dyd, const void* dy, void* cd,
                                                void* dxd, void* dx, int requested,
                                                int* returned, void* results, void* ws,
                                                size_t wsz) {
    (void)h; (void)wd; (void)w; (void)dyd; (void)dy; (void)cd; (void)dxd; (void)dx;
    (void)ws; (void)wsz;
    if (returned) *returned = 1;
    if (results && requested > 0) memset(results, 0, 64);
    return 0;
}

int cudnnConvolutionForward(void* h, const void* alpha, void* xDesc, const void* x,
                            void* wDesc, const void* w, void* convDesc, int algo,
                            void* ws, size_t wsz, const void* beta, void* yDesc,
                            void* y) {
    (void)h; (void)ws; (void)wsz;
    VVA((long)alpha, (long)xDesc, (long)x, (long)wDesc, (long)w, (long)convDesc, algo,
        (long)beta, (long)yDesc, (long)y);
    return (int)vvhost(VVH_CUDNN_CONV_FORWARD, a);
}
int cudnnConvolutionBackwardData(void* h, const void* alpha, void* wDesc,
                                 const void* w, void* dyDesc, const void* dy,
                                 void* convDesc, int algo, void* ws, size_t wsz,
                                 const void* beta, void* dxDesc, void* dx) {
    (void)h; (void)ws; (void)wsz;
    VVA((long)alpha, (long)wDesc, (long)w, (long)dyDesc, (long)dy, (long)convDesc, algo,
        (long)beta, (long)dxDesc, (long)dx);
    return (int)vvhost(VVH_CUDNN_CONV_BACKWARD_DATA, a);
}
int cudnnAddTensor(void* h, const void* alpha, void* aDesc, const void* A,
                   const void* beta, void* cDesc, void* C) {
    (void)h;
    VVA((long)alpha, (long)aDesc, (long)A, (long)beta, (long)cDesc, (long)C);
    return (int)vvhost(VVH_CUDNN_ADD_TENSOR, a);
}
