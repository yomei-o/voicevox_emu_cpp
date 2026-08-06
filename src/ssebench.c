// ssebench.c - how fast does the emulator run the instructions ORT is made of?
//
// The pipeline's cost is dominated by SSE2 kernels, and the emulator implements
// each of them as a scalar loop over sixteen bytes.  A host - x86-64 or
// WebAssembly - has 128-bit vector instructions of its own that map almost one
// to one, so this ought to be the cheapest large speedup available short of a
// JIT.  Before writing any of that, something has to be measurable in seconds
// rather than in the ninety-nine minutes an inference takes.
//
// So: a tight loop over the operations MLAS's SSE2 paths actually use, with a
// checksum printed so nothing can be optimised away, and a fixed instruction
// count so two runs are comparable.
//
//     gcc -O2 -o ssebench ssebench.c        # native: fractions of a second
//     x86emu --sysroot sysroot .../ssebench # emulated: the number that matters
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ROUNDS 2000
#define LANES 256  // 4 KB of 16-byte vectors, small enough to stay in cache

typedef struct {
    uint8_t b[16];
} v128 __attribute__((aligned(16)));

static v128 A[LANES], B[LANES], C[LANES];

static uint64_t fnv(const void* p, size_t n, uint64_t h) {
    const unsigned char* q = p;
    for (size_t i = 0; i < n; i++) h = (h ^ q[i]) * 1099511628211ull;
    return h;
}

// One pass over the arrays for a given instruction.  The operands are reloaded
// every iteration so the emulator sees the memory operand forms too, which is
// what real kernel code looks like.
#define PASS(mnem)                                                            \
    for (int i = 0; i < LANES; i++) {                                         \
        __asm__ volatile("movdqu %1, %%xmm0\n\t"                              \
                         "movdqu %2, %%xmm1\n\t" mnem                         \
                         " %%xmm1, %%xmm0\n\t"                                \
                         "movdqu %%xmm0, %0"                                  \
                         : "=m"(C[i])                                         \
                         : "m"(A[i]), "m"(B[i])                               \
                         : "xmm0", "xmm1", "memory");                         \
    }

int main(void) {
    for (int i = 0; i < LANES; i++)
        for (int j = 0; j < 16; j++) {
            A[i].b[j] = (uint8_t)(i * 7 + j);
            B[i].b[j] = (uint8_t)(i * 13 + j * 3 + 1);
        }

    // The asm blocks are volatile with a memory clobber, so nothing here can be
    // optimised away and the checksum only has to be taken once.  Taking it per
    // pass, as this first did, walks 4 KB fourteen times a round - a hundred
    // million scalar iterations, which buried the thing being measured.
    for (int r = 0; r < ROUNDS; r++) {
        // The integer side: what a quantised or reference kernel is made of.
        PASS("paddb")
        PASS("psubw")
        PASS("pmullw")
        PASS("pmaddwd")
        PASS("pand")
        PASS("pxor")
        PASS("punpcklbw")
        PASS("packsswb")
        // The float side: what the vocoder is made of.
        PASS("addps")
        PASS("mulps")
        PASS("subps")
        PASS("maxps")
        PASS("addpd")
        PASS("mulpd")
    }


    printf("checksum %016llx\n",
           (unsigned long long)fnv(C, sizeof C, 1469598103934665603ull));
    printf("%d SSE instructions executed\n", ROUNDS * LANES * 14);
    return 0;
}
