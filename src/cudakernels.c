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

static const struct {
    const char* key;
    int nargs;
    Handler fn;
} kKernels[] = {
    {"_UnaryElementWiseI", 4, do_unary},
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
