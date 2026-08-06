// Native stand-ins for the ONNX Runtime CUDA kernels a real model launches.
//
// `cudaLaunchKernel` arrives with an opaque function pointer and a `void**` of
// argument addresses - and *no count*, which is why a generic dump walks off
// the end.  The count comes from the kernel's signature, and the signature is
// recoverable from its mangled name, which `__cudaRegisterFunction` handed us.
// So: a table, keyed by the distinctive part of the mangled name, giving the
// number of arguments and what to do with them.
//
// The signatures below were read off the binary with c++filt.  For example
//
//   _ZN11onnxruntime4cuda17_UnaryElementWiseIffNS0_7OP_TanhIfEELi256ELi4EEE
//       vPKT_PT0_T1_i
//
// demangles to
//
//   void _UnaryElementWise<float, float, OP_Tanh<float>, 256, 4>(
//            float const*, float*, OP_Tanh<float>, int)
//
// - four arguments, the third of which is an empty functor.  Every family here
// is one loop shape parameterised that way, which is what makes this tractable
// at all: twenty shapes and a dozen scalar functions, not 4757 kernels.
#include "cudastub.h"

#include <math.h>
#include <string.h>

// ---------------------------------------------------------------------------
// The structs ONNX Runtime passes by value.  Neither is public API, so the
// layouts here are the ones observed in this binary (see VVSTUB_ARGS) rather
// than copied from a source tree that may not match.

// Integer division by a constant, done with a multiply.  Only the divisor is
// needed to reproduce the arithmetic - the magic number is an optimisation.
typedef struct {
    int d_;         // divisor
    unsigned M_;    // magic multiplier
    unsigned l_;    // shift
} DivMod;

// A small fixed-capacity array passed by value: a size and the elements.
typedef struct {
    int size_;
    long long data_[8];
} TArrayLong;

// ---------------------------------------------------------------------------

static int arg_dump;  // VVSTUB_KARGS=1: print each handled kernel's arguments

static void dump(const char* what, void** args, int n) {
    if (!arg_dump) return;
    printf("      %s, %d args\n", what, n);
    for (int i = 0; i < n; i++) {
        const unsigned char* p = args[i];
        printf("        [%d]", i);
        for (int b = 0; b < 24; b++) printf(" %02x", p[b]);
        printf("\n");
    }
    fflush(stdout);
}

#define ARG(type, i) (*(type*)args[i])

// ---- the element-wise families --------------------------------------------
// One loop shape, a different scalar function each time.  ORT's own names for
// them are in the mangled string, which is how the right one gets picked.

typedef float (*UnaryOp)(float);

static float op_identity(float x) { return x; }
static float op_relu(float x) { return x > 0.0f ? x : 0.0f; }
static float op_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float op_tanh(float x) { return tanhf(x); }
static float op_sqrt(float x) { return sqrtf(x); }
static float op_sin(float x) { return sinf(x); }
static float op_cos(float x) { return cosf(x); }
static float op_exp(float x) { return expf(x); }
static float op_log(float x) { return logf(x); }
static float op_neg(float x) { return -x; }
static float op_abs(float x) { return fabsf(x); }
static float op_reciprocal(float x) { return 1.0f / x; }
// x * sigmoid(alpha * x); ORT's default alpha is 1.702, and the functor carries
// it, so it is read from the argument rather than assumed.
static float op_quickgelu_alpha = 1.702f;
static float op_quickgelu(float x) {
    return x / (1.0f + expf(-op_quickgelu_alpha * x));
}
// LeakyRelu's slope likewise travels in the functor.
static float op_leakyrelu_alpha = 0.01f;
static float op_leakyrelu(float x) { return x >= 0.0f ? x : op_leakyrelu_alpha * x; }

static UnaryOp unary_op_for(const char* name) {
    if (strstr(name, "OP_QuickGelu")) return op_quickgelu;
    if (strstr(name, "OP_LeakyRelu")) return op_leakyrelu;
    if (strstr(name, "OP_Sigmoid")) return op_sigmoid;
    if (strstr(name, "OP_Tanh")) return op_tanh;
    if (strstr(name, "OP_Sqrt")) return op_sqrt;
    if (strstr(name, "OP_Relu")) return op_relu;
    if (strstr(name, "OP_Sin")) return op_sin;
    if (strstr(name, "OP_Cos")) return op_cos;
    if (strstr(name, "OP_Exp")) return op_exp;
    if (strstr(name, "OP_Log")) return op_log;
    if (strstr(name, "OP_Neg")) return op_neg;
    if (strstr(name, "OP_Abs")) return op_abs;
    if (strstr(name, "OP_Reciprocal")) return op_reciprocal;
    if (strstr(name, "OP_Div")) return op_reciprocal;  // unary Div is 1/x
    if (strstr(name, "OP_Identity") || strstr(name, "OP_Cast")) return op_identity;
    return NULL;
}

// void _UnaryElementWise<float, float, OP, 256, 4>(const float*, float*, OP, int)
static int do_unary(const char* name, void** args) {
    UnaryOp op = unary_op_for(name);
    if (!op) return 0;
    const float* in = ARG(const float*, 0);
    float* out = ARG(float*, 1);
    // The functor is passed by value; the ones that carry a parameter carry it
    // as their only member, at offset zero.
    if (op == op_quickgelu) memcpy(&op_quickgelu_alpha, args[2], sizeof(float));
    if (op == op_leakyrelu) memcpy(&op_leakyrelu_alpha, args[2], sizeof(float));
    int n = ARG(int, 3);
    for (int i = 0; i < n; i++) out[i] = op(in[i]);
    return 1;
}

// ---------------------------------------------------------------------------
// The table.  A kernel is matched by the distinctive part of its mangled name,
// which is stable across the type specialisations that share a shape.

typedef int (*Handler)(const char* name, void** args);

// Not implemented yet, but listed so their arguments can be dumped safely:
// `cudaLaunchKernel` gives no argument count, so walking the array without one
// reads past the end and segfaults.  The count comes from the signature.
static int do_dump_only(const char* name, void** args) {
    (void)name;
    (void)args;
    return 0;
}

// The divisor is the first field, and it is the only one needed: `M_` and `l_`
// are a multiply-and-shift standing in for the division, which plain arithmetic
// does just as correctly and rather more legibly.
static void divmod(const DivMod* f, int n, int* q, int* r) {
    *q = n / f->d_;
    *r = n - *q * f->d_;
}

// void ExpandKernel2D<T>(int N, const T* in, T* out, DivMod<int> fdm_out_stride0,
//                        int in_view_stride0, int in_view_stride1)
// Each output position splits into (row, column) by the output's row stride,
// and reads through the input's own strides - which are zero along whichever
// axis is being expanded.
static int do_expand2d(const char* name, void** args) {
    if (!strstr(name, "ExpandKernel2DIi") && !strstr(name, "ExpandKernel2DIl")) return 0;
    int is64 = strstr(name, "ExpandKernel2DIl") != NULL;
    int n = ARG(int, 0);
    const void* in = ARG(const void*, 1);
    void* out = ARG(void*, 2);
    const DivMod* fdm = (const DivMod*)args[3];
    int s0 = ARG(int, 4), s1 = ARG(int, 5);
    for (int i = 0; i < n; i++) {
        int r, c;
        divmod(fdm, i, &r, &c);
        long src = (long)r * s0 + (long)c * s1;
        if (is64)
            ((long long*)out)[i] = ((const long long*)in)[src];
        else
            ((int*)out)[i] = ((const int*)in)[src];
    }
    return 1;
}

// An index array is int32 or int64 depending on what the graph declared, and
// the kernel is told which by the element size rather than by its type.
static long long index_at(const void* data, unsigned long elem_size, int i) {
    if (elem_size == 4) return ((const int*)data)[i];
    return ((const long long*)data)[i];
}

// void _GatherKernel<T>(long input_block_size, long indices_max,
//                       DivMod<int> output_block_size, DivMod<int> block_size,
//                       const void* indices, unsigned long index_element_size,
//                       const T* input, T* output, int N)
//
// Every output position decomposes twice: once to find which block of the input
// it belongs to, and once to split that block into "which index" and "where
// within the indexed row".  A negative index counts from the end, and one that
// is still out of range writes a zero rather than reading wild.
static int do_gather(const char* name, void** args) {
    int width = strstr(name, "_GatherKernelIi") ? 4
              : strstr(name, "_GatherKernelIl") ? 8
              : strstr(name, "_GatherKernelIf") ? 4
              : 0;
    if (!width) return 0;
    long long input_block_size = ARG(long long, 0);
    long long indices_max = ARG(long long, 1);
    const DivMod* out_block = (const DivMod*)args[2];
    const DivMod* block = (const DivMod*)args[3];
    const void* indices = ARG(const void*, 4);
    unsigned long index_size = ARG(unsigned long, 5);
    const char* in = ARG(const char*, 6);
    char* out = ARG(char*, 7);
    int n = ARG(int, 8);

    for (int id = 0; id < n; id++) {
        int input_block_index, block_offset;
        divmod(out_block, id, &input_block_index, &block_offset);
        int indices_index, offset;
        divmod(block, block_offset, &indices_index, &offset);
        long long idx = index_at(indices, index_size, indices_index);
        if (idx < 0) idx += indices_max;
        if (idx < 0 || idx >= indices_max) {
            memset(out + (long)id * width, 0, width);
            continue;
        }
        long long src = input_block_index * input_block_size + idx * block->d_ + offset;
        memcpy(out + (long)id * width, in + src * width, width);
    }
    return 1;
}

// void _ConcatKernel<T>(DivMod<int> block_size_including_axis_dim,
//                       DivMod<int> block_size_inside_axis_dim,
//                       const long* concat_sizes, const long* concat_sizes_range,
//                       const long* axis_dimension_input_output_mapping,
//                       T* output, const void** inputs, int N)
//
// The same decomposition, but the middle index says which *input* the position
// came from, and the running total says how far into that input it is.
static int do_concat(const char* name, void** args) {
    int width = strstr(name, "_ConcatKernelIi") ? 4
              : strstr(name, "_ConcatKernelIl") ? 8
              : strstr(name, "_ConcatKernelIf") ? 4
              : 0;
    if (!width) return 0;
    const DivMod* outer = (const DivMod*)args[0];
    const DivMod* inside = (const DivMod*)args[1];
    const long long* concat_sizes = ARG(const long long*, 2);
    const long long* concat_range = ARG(const long long*, 3);
    const long long* mapping = ARG(const long long*, 4);
    char* out = ARG(char*, 5);
    const void** inputs = ARG(const void**, 6);
    int n = ARG(int, 7);

    for (int id = 0; id < n; id++) {
        int outer_block_index, rest;
        divmod(outer, id, &outer_block_index, &rest);
        int block_index, offset;
        divmod(inside, rest, &block_index, &offset);
        int input_index = (int)mapping[block_index];
        long long range_left = input_index == 0 ? 0 : concat_range[input_index - 1];
        long long block_offset = block_index - range_left;
        const char* in = (const char*)inputs[input_index];
        long long src = (long long)outer_block_index * concat_sizes[input_index] * inside->d_ +
                        block_offset * inside->d_ + offset;
        memcpy(out + (long)id * width, in + src * width, width);
    }
    return 1;
}

static const struct {
    const char* key;
    int nargs;
    Handler fn;
} kKernels[] = {
    {"_UnaryElementWiseI", 4, do_unary},
    // void ExpandKernel2D<T>(int, const T*, T*, DivMod<int>, int, int)
    {"ExpandKernel2DI", 6, do_expand2d},
    // void _GatherKernel<T>(long, long, DivMod<int>, DivMod<int>, const void*,
    //                       unsigned long, const T*, T*, int)
    {"_GatherKernelI", 9, do_gather},
    // void _ConcatKernel<T>(DivMod<int>, DivMod<int>, const long*, const long*,
    //                       const long*, T*, const void**, int)
    {"_ConcatKernelI", 8, do_concat},
};

// Returns 1 when the launch was handled, 0 when nothing here knows it yet.
int vvstub_run_kernel(const char* name, void** args) {
    static int inited;
    if (!inited) {
        const char* a = getenv("VVSTUB_KARGS");
        arg_dump = a && *a && *a != '0';
        inited = 1;
    }
    for (size_t i = 0; i < sizeof kKernels / sizeof kKernels[0]; i++) {
        if (!strstr(name, kKernels[i].key)) continue;
        dump(kKernels[i].key, args, kKernels[i].nargs);
        return kKernels[i].fn(name, args);
    }
    return 0;
}
