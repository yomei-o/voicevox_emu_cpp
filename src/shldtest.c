// Every SHLD/SHRD case, printed, so a diff names the operands rather than a
// checksum.  Temporary scaffolding for the isatest disagreement.
#include <stdint.h>
#include <stdio.h>

static const uint64_t G[] = {
    0x0000000000000000ull, 0x0000000000000001ull, 0x00000000ffffffffull,
    0xffffffffffffffffull, 0x8000000000000000ull, 0x7fffffffffffffffull,
    0x0123456789abcdefull, 0xfedcba9876543210ull, 0x00000000000000ffull,
    0x000000000000ff00ull, 0x5555555555555555ull, 0xaaaaaaaaaaaaaaaaull,
    0x000000007fffffffull, 0x0000000080000000ull,
};
#define NG (int)(sizeof G / sizeof G[0])

int main(void) {
    for (int i = 0; i < NG; i++)
        for (int j = 0; j < NG; j++)
            for (int n = 0; n < 72; n += 3) {
                uint64_t a = G[i], f, cnt = (uint64_t)n;
                __asm__ volatile("movq %5, %%rcx\n\t"
                                 "pushq $2\n\t"
                                 "popfq\n\t"
                                 "shldl %%cl, %k3, %k0\n\t"
                                 "pushfq\n\t"
                                 "popq %1"
                                 : "+r"(a), "=&r"(f)
                                 : "0"(a), "r"(G[j]), "r"(cnt), "r"(cnt)
                                 : "cc", "rcx");
                uint64_t b = G[i], g;
                __asm__ volatile("movq %5, %%rcx\n\t"
                                 "pushq $2\n\t"
                                 "popfq\n\t"
                                 "shrdl %%cl, %k3, %k0\n\t"
                                 "pushfq\n\t"
                                 "popq %1"
                                 : "+r"(b), "=&r"(g)
                                 : "0"(b), "r"(G[j]), "r"(cnt), "r"(cnt)
                                 : "cc", "rcx");
                printf("%2d %2d %2d  shld %016llx f=%03llx  shrd %016llx f=%03llx\n", i, j, n,
                       (unsigned long long)a, (unsigned long long)(f & 0x8d5),
                       (unsigned long long)b, (unsigned long long)(g & 0x8d5));
            }
    return 0;
}
