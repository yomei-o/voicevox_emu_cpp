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
#include <stdlib.h>
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

// A small fixed-capacity array passed by value: a count and the elements.  The
// count comes first, so where `data_` begins depends on the element's alignment
// - eight bytes in for a 64-bit element, four for a DivMod.  Letting the C
// compiler lay these out gets that right without a table of offsets.
typedef struct {
    int size_;
    long long data_[8];
} TArrayLong;

typedef struct {
    int size_;
    DivMod data_[8];
} TArrayDivMod;

// A pointer the way the *guest* stores one: eight bytes, whatever the host is.
//
// This matters wherever an array of pointers is indexed rather than read one at
// a time.  A single pointer arrives in an eight-byte argument slot and reading
// four bytes of it on a 32-bit host still gets the right value, little-endian -
// but `inputs[1]` with a four-byte stride reads the *top half of inputs[0]*,
// which is zero.  On x86-64 the two are the same and the mistake is invisible;
// in WebAssembly it is the difference between concatenating two tensors and
// concatenating one of them with a null.
typedef unsigned long long GuestPtr;

static void* guest_ptr(GuestPtr p) { return (void*)(size_t)p; }

typedef struct {
    int size_;
    GuestPtr data_[32];
} TArrayPtr;

// ---------------------------------------------------------------------------

// The divisor is the first field, and it is the only one needed: `M_` and `l_`
// are a multiply-and-shift standing in for the division, which plain arithmetic
// does just as correctly and rather more legibly.
static void divmod(const DivMod* f, int n, int* q, int* r) {
    *q = n / f->d_;
    *r = n - *q * f->d_;
}

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

// ---------------------------------------------------------------------------
// VVSTUB_STATS=1: after each handled kernel, the range of what it wrote.
//
// A wrong loop shape rarely announces itself - it produces numbers, and the
// numbers are wrong somewhere downstream.  Watching each kernel's output range
// is what turns "the audio is silence" into "this kernel is where it left the
// rails".

static int stats_on;
static const void* out_ptr;
static int out_type;
static long long out_n;

#define NOTE_OUT(p, t, n) \
    do { out_ptr = (const void*)(p); out_type = (t); out_n = (n); } while (0)

static void report(const char* key);  // defined once the loaders below exist

// ---------------------------------------------------------------------------
// Reading the specialisation out of the mangled name.
//
// Every family here is one loop shape over whatever element type the graph
// declared, and the type is a single letter in the mangled name.  Parsing that
// letter once is what keeps this to one handler per shape rather than one per
// (shape, type) pair.

typedef enum { T_F32, T_I64, T_I32, T_U32, T_BOOL, T_BAD } ElemType;

static ElemType type_of_char(char c) {
    switch (c) {
        case 'f': return T_F32;
        case 'l': case 'x': return T_I64;
        case 'i': return T_I32;
        case 'j': return T_U32;
        case 'b': return T_BOOL;
        default:  return T_BAD;
    }
}

static int width_of(ElemType t) {
    switch (t) {
        case T_F32: case T_I32: case T_U32: return 4;
        case T_I64: return 8;
        case T_BOOL: return 1;
        default: return 0;
    }
}

// Skip the `Lb0E` / `Li256E` literal template arguments that can sit between
// the opening `I` and the first type letter.
static const char* skip_literals(const char* p) {
    while (*p == 'L') {
        const char* e = strchr(p, 'E');
        if (!e) return p;
        p = e + 1;
    }
    return p;
}

// The first type letter of a specialisation, given the key it was matched on.
static ElemType elem_type_after(const char* name, const char* key) {
    const char* p = strstr(name, key);
    if (!p) return T_BAD;
    p += strlen(key);
    if (*p == 'I') p++;
    return type_of_char(*skip_literals(p));
}

// The `which`th literal template argument, e.g. the `2` of
// `_SliceKernelILb0ELi2EiE` at which = 1.
static int literal_int_after(const char* name, const char* key, int which) {
    const char* p = strstr(name, key);
    if (!p) return -1;
    p += strlen(key);
    if (*p == 'I') p++;
    for (;;) {
        if (*p != 'L') return -1;
        const char* e = strchr(p, 'E');
        if (!e) return -1;
        if (which-- == 0) {
            const char* d = p + 1;
            while (d < e && (*d < '0' || *d > '9')) d++;
            return (int)strtol(d, NULL, 10);
        }
        p = e + 1;
    }
}

// The last `Lb0E` / `Lb1E` in a mangled name: the trailing boolean template
// argument that these kernels use for a mode flag.  It is the *last* one
// because it sits after the type arguments, which can contain `Lb` of their own.
static int last_bool(const char* name) {
    const char* found = NULL;
    for (const char* p = name; (p = strstr(p, "Lb")) != NULL; p++) found = p;
    return found ? found[2] - '0' : 0;
}

static float load_f(const void* p, ElemType t, long long i) {
    switch (t) {
        case T_F32:  return ((const float*)p)[i];
        case T_I64:  return (float)((const long long*)p)[i];
        case T_I32:  return (float)((const int*)p)[i];
        case T_U32:  return (float)((const unsigned*)p)[i];
        case T_BOOL: return ((const unsigned char*)p)[i] ? 1.0f : 0.0f;
        default:     return 0.0f;
    }
}

static long long load_i(const void* p, ElemType t, long long i) {
    switch (t) {
        case T_F32:  return (long long)((const float*)p)[i];
        case T_I64:  return ((const long long*)p)[i];
        case T_I32:  return ((const int*)p)[i];
        case T_U32:  return ((const unsigned*)p)[i];
        case T_BOOL: return ((const unsigned char*)p)[i] ? 1 : 0;
        default:     return 0;
    }
}

static void store_f(void* p, ElemType t, long long i, float v) {
    switch (t) {
        case T_F32:  ((float*)p)[i] = v; break;
        case T_I64:  ((long long*)p)[i] = (long long)v; break;
        case T_I32:  ((int*)p)[i] = (int)v; break;
        case T_U32:  ((unsigned*)p)[i] = (unsigned)v; break;
        case T_BOOL: ((unsigned char*)p)[i] = v != 0.0f; break;
        default: break;
    }
}

static void store_i(void* p, ElemType t, long long i, long long v) {
    switch (t) {
        case T_F32:  ((float*)p)[i] = (float)v; break;
        case T_I64:  ((long long*)p)[i] = v; break;
        case T_I32:  ((int*)p)[i] = (int)v; break;
        case T_U32:  ((unsigned*)p)[i] = (unsigned)v; break;
        case T_BOOL: ((unsigned char*)p)[i] = v != 0; break;
        default: break;
    }
}

static void report(const char* key) {
    if (!stats_on || !out_ptr || out_n <= 0) return;
    double lo = 0, hi = 0, sum = 0;
    int finite = 1;
    for (long long i = 0; i < out_n; i++) {
        double v = load_f(out_ptr, (ElemType)out_type, i);
        if (v != v) finite = 0;
        if (i == 0 || v < lo) lo = v;
        if (i == 0 || v > hi) hi = v;
        sum += v;
    }
    fprintf(stderr, "[k] %-40s n=%-8lld min=%-12g max=%-12g mean=%-12g%s\n", key,
            out_n, lo, hi, sum / (double)out_n, finite ? "" : "  NaN");
}

// ---- the unary family ------------------------------------------------------
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
// The *unary* Div is how ORT finishes a ReduceMean: the reduction sums, and
// this divides by how many were summed.  The count is in the functor, so it is
// read rather than inferred - and it is not `1/x`, which is what this was until
// the numbers said otherwise.  There are 22 of these and 22 reductions.
static float op_div_by = 1.0f;
static float op_div(float x) { return x / op_div_by; }

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
    if (strstr(name, "OP_Div")) return op_div;
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
    if (op == op_div) memcpy(&op_div_by, args[2], sizeof(float));
    int n = ARG(int, 3);
    for (int i = 0; i < n; i++) out[i] = op(in[i]);
    NOTE_OUT(out, T_F32, n);
    return 1;
}

// ---- the binary families ---------------------------------------------------

typedef enum { B_ADD, B_SUB, B_MUL, B_DIV, B_POW, B_EQUAL, B_BAD } BinOp;

static BinOp binop_of(const char* name) {
    if (strstr(name, "OP_Add"))   return B_ADD;
    if (strstr(name, "OP_Sub"))   return B_SUB;
    if (strstr(name, "OP_Mul"))   return B_MUL;
    if (strstr(name, "OP_Div"))   return B_DIV;
    if (strstr(name, "OP_Pow"))   return B_POW;
    if (strstr(name, "OP_Equal")) return B_EQUAL;
    return B_BAD;
}

// The three element types of a binary kernel, in the order the mangled name
// gives them: output, left, right.  They are not always the same - Equal takes
// two floats and answers a bool.
typedef struct {
    ElemType out, lhs, rhs;
} BinTypes;

static BinTypes bin_types(const char* name, const char* key) {
    BinTypes t = {T_BAD, T_BAD, T_BAD};
    const char* p = strstr(name, key);
    if (!p) return t;
    p += strlen(key);
    if (*p == 'I') p++;
    p = skip_literals(p);
    t.out = type_of_char(p[0]);
    t.lhs = type_of_char(p[1]);
    t.rhs = type_of_char(p[2]);
    return t;
}

// One output element.  Integers stay integers: these carry shapes, where a
// round trip through float would be silently wrong past 2^24.
static void bin_apply(BinOp op, BinTypes t, void* out, long long oi, const void* lhs,
                      long long li, const void* rhs, long long ri) {
    if (t.out == T_BOOL) {
        float a = load_f(lhs, t.lhs, li), b = load_f(rhs, t.rhs, ri);
        ((unsigned char*)out)[oi] = (unsigned char)(op == B_EQUAL ? (a == b) : 0);
        return;
    }
    if (t.out != T_F32 && t.lhs != T_F32 && t.rhs != T_F32) {
        long long a = load_i(lhs, t.lhs, li), b = load_i(rhs, t.rhs, ri), v = 0;
        switch (op) {
            case B_ADD: v = a + b; break;
            case B_SUB: v = a - b; break;
            case B_MUL: v = a * b; break;
            case B_DIV: v = b ? a / b : 0; break;
            case B_POW: v = (long long)powf((float)a, (float)b); break;
            default: break;
        }
        store_i(out, t.out, oi, v);
        return;
    }
    float a = load_f(lhs, t.lhs, li), b = load_f(rhs, t.rhs, ri), v = 0.0f;
    switch (op) {
        case B_ADD: v = a + b; break;
        case B_SUB: v = a - b; break;
        case B_MUL: v = a * b; break;
        case B_DIV: v = a / b; break;
        case B_POW: v = powf(a, b); break;
        default: break;
    }
    store_f(out, t.out, oi, v);
}

// void _BinaryElementWiseSimple<IncL, IncR, T, T1, T2, OP, 256, 4>(
//         const T1* lhs, const T2* rhs, T* out, OP func, int N)
// The two flags say whether each side is indexed or is a single value broadcast
// over the whole tensor.
static int do_binary_simple(const char* name, void** args) {
    static const char key[] = "_BinaryElementWiseSimpleI";
    BinOp op = binop_of(name);
    BinTypes t = bin_types(name, key);
    if (op == B_BAD || t.out == T_BAD || t.lhs == T_BAD || t.rhs == T_BAD) return 0;
    int inc_l = literal_int_after(name, key, 0);
    int inc_r = literal_int_after(name, key, 1);
    if (inc_l < 0 || inc_r < 0) return 0;
    const void* lhs = ARG(const void*, 0);
    const void* rhs = ARG(const void*, 1);
    void* out = ARG(void*, 2);
    int n = ARG(int, 4);
    for (int i = 0; i < n; i++)
        bin_apply(op, t, out, i, lhs, inc_l ? i : 0, rhs, inc_r ? i : 0);
    NOTE_OUT(out, t.out, n);
    return 1;
}

// void _BinaryElementWiseRhsPerChannelBatch1<T, T1, T2, OP, 256, 4>(
//         const T1* lhs, const T2* rhs, DivMod fdm_H, T* out, OP, int N)
// A per-channel operand for a single batch: the right-hand side is indexed by
// which channel the position falls in, which is the position divided by the
// channel's own extent.
static int do_binary_rhs_batch1(const char* name, void** args) {
    static const char key[] = "_BinaryElementWiseRhsPerChannelBatch1I";
    BinOp op = binop_of(name);
    BinTypes t = bin_types(name, key);
    if (op == B_BAD || t.out == T_BAD) return 0;
    const void* lhs = ARG(const void*, 0);
    const void* rhs = ARG(const void*, 1);
    const DivMod* fdm_h = (const DivMod*)args[2];
    void* out = ARG(void*, 3);
    int n = ARG(int, 5);
    for (int i = 0; i < n; i++) {
        int q, r;
        divmod(fdm_h, i, &q, &r);
        bin_apply(op, t, out, i, lhs, i, rhs, q);
    }
    NOTE_OUT(out, t.out, n);
    return 1;
}

// The same with a batch dimension: the channel index wraps within the batch.
static int do_binary_rhs_batchn(const char* name, void** args) {
    static const char key[] = "_BinaryElementWiseRhsPerChannelBatchNI";
    BinOp op = binop_of(name);
    BinTypes t = bin_types(name, key);
    if (op == B_BAD || t.out == T_BAD) return 0;
    const void* lhs = ARG(const void*, 0);
    const void* rhs = ARG(const void*, 1);
    const DivMod* fdm_h = (const DivMod*)args[2];
    const DivMod* fdm_c = (const DivMod*)args[3];
    void* out = ARG(void*, 4);
    int n = ARG(int, 6);
    for (int i = 0; i < n; i++) {
        int q, r;
        divmod(fdm_h, i, &q, &r);
        int channel, batch;
        divmod(fdm_c, q, &batch, &channel);
        bin_apply(op, t, out, i, lhs, i, rhs, channel);
    }
    NOTE_OUT(out, t.out, n);
    return 1;
}

// void _BinaryElementWise<T, T1, T2, OP, lhs_compute, rhs_compute, 256, 4>(
//         int rank, TArray<long> lhs_strides, const T1* lhs,
//         TArray<long> rhs_strides, const T2* rhs, TArray<DivMod> out_strides,
//         T* out, const OP&, int N)
// The general broadcast: walk the output's strides to recover the coordinate,
// then apply each side's own padded strides - which are zero on the axes it is
// being broadcast along.  A side that needs no computing is read straight.
//
// The functor arrives by reference here rather than by value, which would be a
// problem if any of these carried a parameter.  None do: Add, Sub, Mul and Div
// are empty structs, so the pointer is never followed.
static int do_binary_general(const char* name, void** args) {
    static const char key[] = "_BinaryElementWiseI";
    BinOp op = binop_of(name);
    BinTypes t = bin_types(name, key);
    if (op == B_BAD || t.out == T_BAD) return 0;
    // The two flags sit after the functor, immediately before `Li256ELi4E`.
    const char* tail = strstr(name, "Li256ELi4E");
    if (!tail || tail - name < 8) return 0;
    const char* flags = tail - 8;
    if (flags[0] != 'L' || flags[1] != 'b' || flags[4] != 'L' || flags[5] != 'b') return 0;
    int lhs_compute = flags[2] - '0', rhs_compute = flags[6] - '0';

    int rank = ARG(int, 0);
    const TArrayLong* lhs_strides = (const TArrayLong*)args[1];
    const void* lhs = ARG(const void*, 2);
    const TArrayLong* rhs_strides = (const TArrayLong*)args[3];
    const void* rhs = ARG(const void*, 4);
    const TArrayDivMod* out_strides = (const TArrayDivMod*)args[5];
    void* out = ARG(void*, 6);
    int n = ARG(int, 8);
    if (rank > 8) return 0;

    for (int i = 0; i < n; i++) {
        long long li = lhs_compute ? 0 : i;
        long long ri = rhs_compute ? 0 : i;
        int offset = i;
        for (int dim = 0; dim < rank; dim++) {
            int q, r;
            divmod(&out_strides->data_[dim], offset, &q, &r);
            if (lhs_compute) li += (int)lhs_strides->data_[dim] * q;
            if (rhs_compute) ri += (int)rhs_strides->data_[dim] * q;
            offset = r;
        }
        bin_apply(op, t, out, i, lhs, li, rhs, ri);
    }
    NOTE_OUT(out, t.out, n);
    return 1;
}

// void _TenaryElementWise<T, CondKind, XKind, YKind, 256, 4>(
//         size_t rank, TArray<long> cond_strides, const bool* cond,
//         TArray<long> x_strides, const T* x, TArray<long> y_strides, const T* y,
//         TArray<DivMod> out_strides, T* out, int N)
// ONNX's Where.  Each of the three operands is one of: indexed directly (0), a
// single value (1), or broadcast through its own padded strides (2).
static int do_tenary(const char* name, void** args) {
    enum { KIND_DIRECT = 0, KIND_SCALAR = 1, KIND_COMPUTE = 2 };
    ElemType t = elem_type_after(name, "_TenaryElementWiseI");
    int w = width_of(t);
    if (!w) return 0;

    // `<float, (BroadcastIndexType)2, (BroadcastIndexType)1,
    // (BroadcastIndexType)0>` mangles the first enum with its full name and the
    // other two as substitutions: `LNS0_18BroadcastIndexTypeE2ELS2_1ELS2_0E`.
    const char* p = strstr(name, "BroadcastIndexTypeE");
    if (!p) return 0;
    int kind[3];
    kind[0] = p[strlen("BroadcastIndexTypeE")] - '0';
    const char* q = p;
    for (int i = 1; i < 3; i++) {
        q = strstr(q + 1, "LS");
        if (!q) return 0;
        const char* u = strchr(q, '_');
        if (!u) return 0;
        kind[i] = u[1] - '0';
    }
    for (int i = 0; i < 3; i++)
        if (kind[i] < 0 || kind[i] > 2) return 0;

    long long rank = ARG(long long, 0);
    const TArrayLong* strides[3];
    strides[0] = (const TArrayLong*)args[1];
    const unsigned char* cond = ARG(const unsigned char*, 2);
    strides[1] = (const TArrayLong*)args[3];
    const char* x = ARG(const char*, 4);
    strides[2] = (const TArrayLong*)args[5];
    const char* y = ARG(const char*, 6);
    const TArrayDivMod* out_strides = (const TArrayDivMod*)args[7];
    char* out = ARG(char*, 8);
    int n = ARG(int, 9);
    if (rank > 8) return 0;

    for (int i = 0; i < n; i++) {
        long long idx[3];
        for (int k = 0; k < 3; k++) idx[k] = kind[k] == KIND_DIRECT ? i : 0;
        int offset = i;
        for (int dim = 0; dim < rank; dim++) {
            int qd, r;
            divmod(&out_strides->data_[dim], offset, &qd, &r);
            for (int k = 0; k < 3; k++)
                if (kind[k] == KIND_COMPUTE) idx[k] += (int)strides[k]->data_[dim] * qd;
            offset = r;
        }
        int take_x = cond[idx[0]] != 0;
        const char* from = take_x ? x : y;
        long long src = take_x ? idx[1] : idx[2];
        memcpy(out + (long long)i * w, from + src * w, (size_t)w);
    }
    NOTE_OUT(out, t, n);
    return 1;
}

// ---- the layout families ---------------------------------------------------

// void ExpandKernel2D<T>(int N, const T* in, T* out, DivMod fdm_out_stride0,
//                        int in_view_stride0, int in_view_stride1)
// Each output position splits into (row, column) by the output's row stride,
// and reads through the input's own strides - which are zero along whichever
// axis is being expanded.
static int do_expand2d(const char* name, void** args) {
    ElemType t = elem_type_after(name, "ExpandKernel2D");
    int w = width_of(t);
    if (!w) return 0;
    int n = ARG(int, 0);
    const char* in = ARG(const char*, 1);
    char* out = ARG(char*, 2);
    const DivMod* fdm = (const DivMod*)args[3];
    int s0 = ARG(int, 4), s1 = ARG(int, 5);
    for (int i = 0; i < n; i++) {
        int r, c;
        divmod(fdm, i, &r, &c);
        long long src = (long long)r * s0 + (long long)c * s1;
        memcpy(out + (long long)i * w, in + src * w, (size_t)w);
    }
    NOTE_OUT(out, t, n);
    return 1;
}

// void ExpandKernel<T, 256, 4>(int rank, int N, const T* in, T* out,
//                              TArray<DivMod> out_strides, TArray<long> in_strides)
// The N-dimensional form of the above, and the same idea.
static int do_expand(const char* name, void** args) {
    ElemType t = elem_type_after(name, "ExpandKernel");
    int w = width_of(t);
    if (!w) return 0;
    int rank = ARG(int, 0);
    int n = ARG(int, 1);
    const char* in = ARG(const char*, 2);
    char* out = ARG(char*, 3);
    const TArrayDivMod* out_strides = (const TArrayDivMod*)args[4];
    const TArrayLong* in_strides = (const TArrayLong*)args[5];
    if (rank > 8) return 0;
    for (int i = 0; i < n; i++) {
        long long src = 0;
        int offset = i;
        for (int dim = 0; dim < rank; dim++) {
            int q, r;
            divmod(&out_strides->data_[dim], offset, &q, &r);
            src += in_strides->data_[dim] * q;
            offset = r;
        }
        memcpy(out + (long long)i * w, in + src * w, (size_t)w);
    }
    NOTE_OUT(out, t, n);
    return 1;
}

// An index array is int32 or int64 depending on what the graph declared, and
// the kernel is told which by the element size rather than by its type.
static long long index_at(const void* data, unsigned long elem_size, int i) {
    if (elem_size == 4) return ((const int*)data)[i];
    return ((const long long*)data)[i];
}

// void _GatherKernel<T>(long input_block_size, long indices_max,
//                       DivMod output_block_size, DivMod block_size,
//                       const void* indices, unsigned long index_element_size,
//                       const T* input, T* output, int N)
//
// Every output position decomposes twice: once to find which block of the input
// it belongs to, and once to split that block into "which index" and "where
// within the indexed row".  A negative index counts from the end, and one that
// is still out of range writes a zero rather than reading wild.
static int do_gather(const char* name, void** args) {
    ElemType t = elem_type_after(name, "_GatherKernel");
    int width = width_of(t);
    if (!width) return 0;
    long long input_block_size = ARG(long long, 0);
    long long indices_max = ARG(long long, 1);
    const DivMod* out_block = (const DivMod*)args[2];
    const DivMod* block = (const DivMod*)args[3];
    const void* indices = ARG(const void*, 4);
    unsigned long long index_size = ARG(unsigned long long, 5);
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
            memset(out + (long long)id * width, 0, (size_t)width);
            continue;
        }
        long long src = input_block_index * input_block_size + idx * block->d_ + offset;
        memcpy(out + (long long)id * width, in + src * width, (size_t)width);
    }
    NOTE_OUT(out, t, n);
    return 1;
}

// void _ConcatKernel<T>(DivMod block_size_including_axis_dim,
//                       DivMod block_size_inside_axis_dim,
//                       const long* concat_sizes, const long* concat_sizes_range,
//                       const long* axis_dimension_input_output_mapping,
//                       T* output, const void** inputs, int N)
//
// The same decomposition, but the middle index says which *input* the position
// came from, and the running total says how far into that input it is.
static int do_concat(const char* name, void** args) {
    ElemType t = elem_type_after(name, "_ConcatKernel");
    int width = width_of(t);
    if (!width) return 0;
    const DivMod* outer = (const DivMod*)args[0];
    const DivMod* inside = (const DivMod*)args[1];
    const long long* concat_sizes = ARG(const long long*, 2);
    const long long* concat_range = ARG(const long long*, 3);
    const long long* mapping = ARG(const long long*, 4);
    char* out = ARG(char*, 5);
    const GuestPtr* inputs = ARG(const GuestPtr*, 6);
    int n = ARG(int, 7);

    for (int id = 0; id < n; id++) {
        int outer_block_index, rest;
        divmod(outer, id, &outer_block_index, &rest);
        int block_index, offset;
        divmod(inside, rest, &block_index, &offset);
        int input_index = (int)mapping[block_index];
        long long range_left = input_index == 0 ? 0 : concat_range[input_index - 1];
        long long block_offset = block_index - range_left;
        const char* in = (const char*)guest_ptr(inputs[input_index]);
        long long src = (long long)outer_block_index * concat_sizes[input_index] * inside->d_ +
                        block_offset * inside->d_ + offset;
        memcpy(out + (long long)id * width, in + src * width, (size_t)width);
    }
    NOTE_OUT(out, t, n);
    return 1;
}

// void _ConcatKernelSameConcatDim<T, TArray<const void*, 32>>(
//         DivMod outer, DivMod inside, DivMod concat, T* out,
//         TArray<const void*, 32> inputs, int N)
// The case where every input contributes the same extent along the axis, so
// which input a position came from is a division rather than a table lookup.
static int do_concat_same(const char* name, void** args) {
    ElemType t = elem_type_after(name, "_ConcatKernelSameConcatDim");
    int w = width_of(t);
    if (!w) return 0;
    const DivMod* outer = (const DivMod*)args[0];
    const DivMod* inside = (const DivMod*)args[1];
    const DivMod* concat = (const DivMod*)args[2];
    char* out = ARG(char*, 3);
    const TArrayPtr* inputs = (const TArrayPtr*)args[4];
    int n = ARG(int, 5);
    for (int id = 0; id < n; id++) {
        int outer_index, rest;
        divmod(outer, id, &outer_index, &rest);
        int block_index, offset;
        divmod(inside, rest, &block_index, &offset);
        int input_index, block_offset;
        divmod(concat, block_index, &input_index, &block_offset);
        long long src = (long long)outer_index * concat->d_ * inside->d_ +
                        (long long)block_offset * inside->d_ + offset;
        const char* in = (const char*)guest_ptr(inputs->data_[input_index]);
        memcpy(out + (long long)id * w, in + src * w, (size_t)w);
    }
    NOTE_OUT(out, t, n);
    return 1;
}

// The transpose of the above: one input scattered into equal-sized outputs.
static int do_split_same(const char* name, void** args) {
    ElemType t = elem_type_after(name, "_SplitKernelSameSplitDim");
    int w = width_of(t);
    if (!w) return 0;
    const DivMod* outer = (const DivMod*)args[0];
    const DivMod* inside = (const DivMod*)args[1];
    const DivMod* split = (const DivMod*)args[2];
    const char* in = ARG(const char*, 4);
    const TArrayPtr* outputs = (const TArrayPtr*)args[5];
    int n = ARG(int, 6);
    for (int id = 0; id < n; id++) {
        int outer_index, rest;
        divmod(outer, id, &outer_index, &rest);
        int block_index, offset;
        divmod(inside, rest, &block_index, &offset);
        int output_index, block_offset;
        divmod(split, block_index, &output_index, &block_offset);
        long long dst = (long long)outer_index * split->d_ * inside->d_ +
                        (long long)block_offset * inside->d_ + offset;
        char* o = (char*)guest_ptr(outputs->data_[output_index]);
        memcpy(o + dst * w, in + (long long)id * w, (size_t)w);
    }
    return 1;
}

// void TransposeKernel<T>(int rank, TArray<long> in_strides, const T* in,
//                         TArray<DivMod> out_strides, T* out, int N)
static int do_transpose(const char* name, void** args) {
    ElemType t = elem_type_after(name, "TransposeKernel");
    int w = width_of(t);
    if (!w) return 0;
    int rank = ARG(int, 0);
    const TArrayLong* in_strides = (const TArrayLong*)args[1];
    const char* in = ARG(const char*, 2);
    const TArrayDivMod* out_strides = (const TArrayDivMod*)args[3];
    char* out = ARG(char*, 4);
    int n = ARG(int, 5);
    if (rank > 8) return 0;
    for (int i = 0; i < n; i++) {
        long long src = 0;
        int offset = i;
        for (int dim = 0; dim < rank; dim++) {
            int q, r;
            divmod(&out_strides->data_[dim], offset, &q, &r);
            src += in_strides->data_[dim] * q;
            offset = r;
        }
        memcpy(out + (long long)i * w, in + src * w, (size_t)w);
    }
    NOTE_OUT(out, t, n);
    return 1;
}

// void _SliceKernel<is_grad, DIMS, T>(TArray<long> starts, TArray<long> steps,
//                                     TArray<long> in_strides,
//                                     TArray<DivMod> out_strides,
//                                     const T* in, T* out, int N)
// The rank is a template parameter here rather than an argument, so it comes
// out of the mangled name: `_SliceKernelILb0ELi4EiE` is four dimensions of int.
static int do_slice(const char* name, void** args) {
    static const char key[] = "_SliceKernelI";
    int dims = literal_int_after(name, key, 1);
    ElemType t = elem_type_after(name, key);
    int w = width_of(t);
    if (!w || dims <= 0 || dims > 8) return 0;
    const TArrayLong* starts = (const TArrayLong*)args[0];
    const TArrayLong* steps = (const TArrayLong*)args[1];
    const TArrayLong* in_strides = (const TArrayLong*)args[2];
    const TArrayDivMod* out_strides = (const TArrayDivMod*)args[3];
    const char* in = ARG(const char*, 4);
    char* out = ARG(char*, 5);
    int n = ARG(int, 6);
    for (int i = 0; i < n; i++) {
        long long src = 0;
        int offset = i;
        for (int dim = 0; dim < dims; dim++) {
            int q, r;
            divmod(&out_strides->data_[dim], offset, &q, &r);
            src += (starts->data_[dim] + steps->data_[dim] * q) * in_strides->data_[dim];
            offset = r;
        }
        memcpy(out + (long long)i * w, in + src * w, (size_t)w);
    }
    NOTE_OUT(out, t, n);
    return 1;
}

// void _ScatterNDKernel<T>(T* out, size_t num_indices, const long* indices,
//                          long last_index_dimension,
//                          const long* element_counts_and_input_dims,
//                          const T* updates, size_t num_updates_elements)
// The first half of that trailing array is the stride of each indexed axis, the
// second half its extent.  An index out of range is clamped rather than
// faulted, which is what the GPU kernel does - there is nowhere to throw from.
static int do_scatter_nd(const char* name, void** args) {
    ElemType t = elem_type_after(name, "_ScatterNDKernel");
    int w = width_of(t);
    if (!w) return 0;
    char* out = ARG(char*, 0);
    unsigned long long num_indices = ARG(unsigned long long, 1);
    const long long* indices = ARG(const long long*, 2);
    long long last_dim = ARG(long long, 3);
    const long long* counts_and_dims = ARG(const long long*, 4);
    const char* updates = ARG(const char*, 5);
    unsigned long long slice = ARG(unsigned long long, 6);
    if (last_dim <= 0 || last_dim > 8) return 0;

    for (unsigned long long id = 0; id < num_indices; id++) {
        long long offset = 0;
        for (long long k = 0; k < last_dim; k++) {
            long long index = indices[last_dim * (long long)id + k];
            long long stride = counts_and_dims[k];
            long long extent = counts_and_dims[k + last_dim];
            if (index >= 0) {
                if (index >= extent) index = extent - 1;
            } else {
                index = index < -extent ? 0 : index + extent;
            }
            offset += index * stride;
        }
        memcpy(out + offset * w, updates + (long long)id * slice * w, (size_t)slice * w);
    }
    return 1;
}

// ---- the small ones --------------------------------------------------------

// void _Fill<T, 256, 4>(T* out, T value, int N)
static int do_fill(const char* name, void** args) {
    ElemType t = elem_type_after(name, "_FillI");
    int w = width_of(t);
    if (!w) return 0;
    char* out = ARG(char*, 0);
    int n = ARG(int, 2);
    for (int i = 0; i < n; i++) memcpy(out + (long long)i * w, args[1], (size_t)w);
    NOTE_OUT(out, t, n);
    return 1;
}

// void _FillFromDataPtrKernel<T, 256, 4>(T* out, const T* value, int N)
// One value, read through a pointer rather than passed by value, written
// everywhere.
static int do_fill_from_ptr(const char* name, void** args) {
    ElemType t = elem_type_after(name, "_FillFromDataPtrKernel");
    int w = width_of(t);
    if (!w) return 0;
    char* out = ARG(char*, 0);
    const char* src = ARG(const char*, 1);
    int n = ARG(int, 2);
    for (int i = 0; i < n; i++) memcpy(out + (long long)i * w, src, (size_t)w);
    NOTE_OUT(out, t, n);
    return 1;
}

// void RangeKernel<T>(T start, T delta, int count, T* out)
static int do_range(const char* name, void** args) {
    ElemType t = elem_type_after(name, "RangeKernel");
    if (t == T_BAD) return 0;
    int n = ARG(int, 2);
    void* out = ARG(void*, 3);
    if (t == T_F32) {
        float start = ARG(float, 0), delta = ARG(float, 1);
        for (int i = 0; i < n; i++) ((float*)out)[i] = start + delta * (float)i;
    } else {
        long long start = load_i(args[0], t, 0), delta = load_i(args[1], t, 0);
        for (int i = 0; i < n; i++) store_i(out, t, i, start + delta * i);
    }
    NOTE_OUT(out, t, n);
    return 1;
}

// void reduce_matrix_columns_kernel<TIn, TOut, TBuf, In, Out, DivideBySize>(
//         int m, int n, const TIn* in, TOut* out, TBuf* buffer, int* done_count)
// `m` rows of `n` columns, summed across the columns.  The buffer and the
// counter are how the GPU version combines partial sums across blocks; one pass
// over the rows needs neither.  The trailing flag is what separates ReduceSum
// from ReduceMean.
static int do_reduce_columns(const char* name, void** args) {
    if (!strstr(name, "reduce_matrix_columns_kernelIfff")) return 0;
    int divide = last_bool(name);
    int m = ARG(int, 0), n = ARG(int, 1);
    const float* in = ARG(const float*, 2);
    float* out = ARG(float*, 3);
    for (int i = 0; i < m; i++) {
        const float* row = in + (long long)i * n;
        float sum = 0.0f;
        for (int j = 0; j < n; j++) sum += row[j];
        out[i] = (divide && n) ? sum / (float)n : sum;
    }
    NOTE_OUT(out, T_F32, m);
    return 1;
}

// void softmax_warp_forward<in, out, acc, log2_elements, is_log>(
//         out* dst, const in* src, int batch_count, int stride, int element_count)
static int do_softmax(const char* name, void** args) {
    if (!strstr(name, "softmax_warp_forwardIfff")) return 0;
    // The trailing `Lb0E` / `Lb1E` chooses softmax or log-softmax.
    int is_log = last_bool(name);
    float* dst = ARG(float*, 0);
    const float* src = ARG(const float*, 1);
    int batch = ARG(int, 2), stride = ARG(int, 3), count = ARG(int, 4);
    if (count <= 0) return 1;
    for (int b = 0; b < batch; b++) {
        const float* s = src + (long long)b * stride;
        float* d = dst + (long long)b * stride;
        float max = s[0];
        for (int i = 1; i < count; i++)
            if (s[i] > max) max = s[i];
        float sum = 0.0f;
        for (int i = 0; i < count; i++) sum += expf(s[i] - max);
        for (int i = 0; i < count; i++)
            d[i] = is_log ? (s[i] - max - logf(sum)) : expf(s[i] - max) / sum;
    }
    NOTE_OUT(dst, T_F32, (long long)batch * stride);
    return 1;
}

// ---------------------------------------------------------------------------
// The table.  A kernel is matched by the distinctive part of its mangled name,
// which is stable across the type specialisations that share a shape.  Order
// matters where one key is a prefix of another: `_ConcatKernelSameConcatDim`
// has to be tried before `_ConcatKernel`.

typedef int (*Handler)(const char* name, void** args);

static const struct {
    const char* key;
    int nargs;
    Handler fn;
} kKernels[] = {
    {"_UnaryElementWiseI", 4, do_unary},
    {"_BinaryElementWiseSimpleI", 5, do_binary_simple},
    {"_BinaryElementWiseRhsPerChannelBatch1I", 6, do_binary_rhs_batch1},
    {"_BinaryElementWiseRhsPerChannelBatchNI", 7, do_binary_rhs_batchn},
    {"_BinaryElementWiseI", 9, do_binary_general},
    {"_TenaryElementWiseI", 10, do_tenary},
    // void ExpandKernel2D<T>(int, const T*, T*, DivMod, int, int)
    {"ExpandKernel2DI", 6, do_expand2d},
    {"ExpandKernelI", 6, do_expand},
    // void _GatherKernel<T>(long, long, DivMod, DivMod, const void*,
    //                       unsigned long, const T*, T*, int)
    {"_GatherKernelI", 9, do_gather},
    {"_ConcatKernelSameConcatDimI", 6, do_concat_same},
    // void _ConcatKernel<T>(DivMod, DivMod, const long*, const long*,
    //                       const long*, T*, const void**, int)
    {"_ConcatKernelI", 8, do_concat},
    {"_SplitKernelSameSplitDimI", 7, do_split_same},
    {"_SliceKernelI", 7, do_slice},
    {"TransposeKernelI", 6, do_transpose},
    {"_ScatterNDKernelI", 7, do_scatter_nd},
    {"_FillFromDataPtrKernelI", 3, do_fill_from_ptr},
    {"_FillI", 3, do_fill},
    {"RangeKernelI", 4, do_range},
    {"reduce_matrix_columns_kernelI", 6, do_reduce_columns},
    {"softmax_warp_forwardI", 5, do_softmax},
};

// How many arguments a kernel takes, or 0 for one this build does not know.
//
// `cudaLaunchKernel` is handed a `void**` with no count, so a caller that has
// to marshal the arguments across a boundary - the emulator's host shim does -
// has no way to know where the array ends.  The count comes from the signature,
// which is what the table already records.
int vvstub_kernel_nargs(const char* name) {
    for (size_t i = 0; i < sizeof kKernels / sizeof kKernels[0]; i++)
        if (strstr(name, kKernels[i].key)) return kKernels[i].nargs;
    return 0;
}

// Returns 1 when the launch was handled, 0 when nothing here knows it yet.
int vvstub_run_kernel(const char* name, void** args) {
    static int inited;
    if (!inited) {
        const char* a = getenv("VVSTUB_KARGS");
        arg_dump = a && *a && *a != '0';
        const char* v = getenv("VVSTUB_STATS");
        stats_on = v && *v && *v != '0';
        inited = 1;
    }
    for (size_t i = 0; i < sizeof kKernels / sizeof kKernels[0]; i++) {
        if (!strstr(name, kKernels[i].key)) continue;
        dump(kKernels[i].key, args, kKernels[i].nargs);
        out_ptr = NULL;
        int handled = kKernels[i].fn(name, args);
        if (handled) report(kKernels[i].key);
        return handled;
    }
    return 0;
}
