// AES-NI against the FIPS-197 known-answer vectors.
//
// Worth its own test because a wrong AES round does not crash: it produces
// plausible bytes that are simply not the right ones, and the caller reports
// something unrelated a long way downstream ("protobuf parsing failed", in the
// case that prompted this).  The published vectors are the only way to know.
//
//   cc -maes -O2 -o aesni aesni.c
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <wmmintrin.h>

static void show(const char* label, const void* p) {
    const unsigned char* b = (const unsigned char*)p;
    printf("%s ", label);
    for (int i = 0; i < 16; i++) printf("%02x", b[i]);
    printf("\n");
}

// FIPS-197 A.1: expand a 128-bit key into 11 round keys, which is what
// AESKEYGENASSIST exists for.
#define EXPAND(rcon)                                                       \
    do {                                                                   \
        __m128i t = _mm_aeskeygenassist_si128(k, rcon);                    \
        t = _mm_shuffle_epi32(t, 0xFF);                                    \
        k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                        \
        k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                        \
        k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                        \
        k = _mm_xor_si128(k, t);                                           \
        rk[n++] = k;                                                       \
    } while (0)

int main(void) {
    const unsigned char key[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                   0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
    const unsigned char plain[16] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                                     0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    const unsigned char expect[16] = {0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30,
                                      0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a};

    __m128i rk[11], k = _mm_loadu_si128((const __m128i*)key);
    int n = 0;
    rk[n++] = k;
    EXPAND(0x01); EXPAND(0x02); EXPAND(0x04); EXPAND(0x08); EXPAND(0x10);
    EXPAND(0x20); EXPAND(0x40); EXPAND(0x80); EXPAND(0x1B); EXPAND(0x36);

    // Encrypt: AESENC nine times, AESENCLAST once.
    __m128i s = _mm_xor_si128(_mm_loadu_si128((const __m128i*)plain), rk[0]);
    for (int i = 1; i < 10; i++) s = _mm_aesenc_si128(s, rk[i]);
    s = _mm_aesenclast_si128(s, rk[10]);

    unsigned char got[16];
    _mm_storeu_si128((__m128i*)got, s);
    show("cipher  ", got);
    show("expect  ", expect);
    int ok = memcmp(got, expect, 16) == 0;
    printf("encrypt %s\n", ok ? "ok" : "WRONG");

    // Decrypt with the equivalent-inverse schedule, which is what AESIMC is
    // for: the round keys run backwards, with InvMixColumns applied to the
    // middle nine.
    __m128i dk[11];
    dk[0] = rk[10];
    for (int i = 1; i < 10; i++) dk[i] = _mm_aesimc_si128(rk[10 - i]);
    dk[10] = rk[0];

    __m128i d = _mm_xor_si128(s, dk[0]);
    for (int i = 1; i < 10; i++) d = _mm_aesdec_si128(d, dk[i]);
    d = _mm_aesdeclast_si128(d, dk[10]);

    unsigned char back[16];
    _mm_storeu_si128((__m128i*)back, d);
    show("plain   ", back);
    int ok2 = memcmp(back, plain, 16) == 0;
    printf("decrypt %s\n", ok2 ? "ok" : "WRONG");

    // PCLMULQDQ, the other instruction a crypto library reaches for.  Squaring
    // is the clearest case: with no carries, every cross term appears twice and
    // cancels, so the bits of the input simply spread out to twice their
    // spacing.  0x11...11 (a bit every 4) squares to a bit every 8: 0x01 in
    // every byte.
    __m128i a = _mm_set_epi64x(0, 0x1111111111111111ull);
    unsigned char cl[16];
    _mm_storeu_si128((__m128i*)cl, _mm_clmulepi64_si128(a, a, 0x00));
    show("clmul   ", cl);
    int ok3 = memcmp(cl, "\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01", 16) == 0;
    printf("clmul   %s\n", ok3 ? "ok" : "WRONG");

    printf("%s\n", (ok && ok2 && ok3) ? "AES OK" : "AES FAILED");
    return (ok && ok2 && ok3) ? 0 : 1;
}
