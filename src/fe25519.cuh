// Field arithmetic over GF(2^255 - 19), 5x51-bit limbs.
// All functions are __host__ __device__ so the exact same code runs on GPU
// (search) and CPU (verification of found keys).
#pragma once
#include <stdint.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

typedef uint64_t fe[5];

#define FE_M51 0x7FFFFFFFFFFFFULL /* 2^51 - 1 */

struct u128s {
    uint64_t lo, hi;
};

__host__ __device__ inline void mul64wide(uint64_t a, uint64_t b, uint64_t* lo, uint64_t* hi)
{
#if defined(__CUDA_ARCH__)
    *lo = a * b;
    *hi = __umul64hi(a, b);
#elif defined(_MSC_VER)
    *lo = _umul128(a, b, hi);
#else
    unsigned __int128 p = (unsigned __int128)a * b;
    *lo = (uint64_t)p;
    *hi = (uint64_t)(p >> 64);
#endif
}

/* t += a*b (128-bit accumulate) */
__host__ __device__ inline void fe_mac(u128s* t, uint64_t a, uint64_t b)
{
#if defined(__CUDA_ARCH__)
    asm("mad.lo.cc.u64 %0, %2, %3, %0;\n\t"
        "madc.hi.u64 %1, %2, %3, %1;"
        : "+l"(t->lo), "+l"(t->hi)
        : "l"(a), "l"(b));
#else
    uint64_t lo, hi;
    mul64wide(a, b, &lo, &hi);
    t->lo += lo;
    t->hi += hi + (t->lo < lo ? 1u : 0u);
#endif
}

/* t += c (64-bit into 128-bit accumulator) */
__host__ __device__ inline void fe_cadd(u128s* t, uint64_t c)
{
#if defined(__CUDA_ARCH__)
    asm("add.cc.u64 %0, %0, %2;\n\t"
        "addc.u64 %1, %1, 0;"
        : "+l"(t->lo), "+l"(t->hi)
        : "l"(c));
#else
    t->lo += c;
    t->hi += (t->lo < c ? 1u : 0u);
#endif
}

__host__ __device__ inline uint64_t fe_shr51(u128s t)
{
    return (t.hi << 13) | (t.lo >> 51);
}

__host__ __device__ inline void fe_0(fe r)
{
    r[0] = r[1] = r[2] = r[3] = r[4] = 0;
}

__host__ __device__ inline void fe_1(fe r)
{
    r[0] = 1;
    r[1] = r[2] = r[3] = r[4] = 0;
}

__host__ __device__ inline void fe_copy(fe r, const fe a)
{
    for (int i = 0; i < 5; i++)
        r[i] = a[i];
}

/* weak reduction: brings all limbs below 2^51 + eps; value preserved mod p.
   Input limbs must be < 2^63 - ish (always the case here). */
__host__ __device__ inline void fe_carry(fe r)
{
    uint64_t c;
    c = r[0] >> 51; r[0] &= FE_M51; r[1] += c;
    c = r[1] >> 51; r[1] &= FE_M51; r[2] += c;
    c = r[2] >> 51; r[2] &= FE_M51; r[3] += c;
    c = r[3] >> 51; r[3] &= FE_M51; r[4] += c;
    c = r[4] >> 51; r[4] &= FE_M51; r[0] += c * 19;
    c = r[0] >> 51; r[0] &= FE_M51; r[1] += c;
}

__host__ __device__ inline void fe_add(fe r, const fe a, const fe b)
{
    for (int i = 0; i < 5; i++)
        r[i] = a[i] + b[i];
    fe_carry(r);
}

/* r = a - b, computed as a + 2p - b to stay non-negative */
__host__ __device__ inline void fe_sub(fe r, const fe a, const fe b)
{
    r[0] = a[0] + 0xFFFFFFFFFFFDAULL - b[0]; /* 2*(2^51-19) */
    r[1] = a[1] + 0xFFFFFFFFFFFFEULL - b[1]; /* 2*(2^51-1)  */
    r[2] = a[2] + 0xFFFFFFFFFFFFEULL - b[2];
    r[3] = a[3] + 0xFFFFFFFFFFFFEULL - b[3];
    r[4] = a[4] + 0xFFFFFFFFFFFFEULL - b[4];
    fe_carry(r);
}

/* no-carry variants: results may reach ~2^54 per limb, which is still a safe
   input to fe_mul/fe_sq (their 128-bit accumulators tolerate limbs < 2^54).
   Use only when the result feeds a multiply, never another no-carry add/sub. */
__host__ __device__ inline void fe_add_nc(fe r, const fe a, const fe b)
{
    for (int i = 0; i < 5; i++)
        r[i] = a[i] + b[i];
}

__host__ __device__ inline void fe_sub_nc(fe r, const fe a, const fe b)
{
    r[0] = a[0] + 0xFFFFFFFFFFFDAULL - b[0];
    r[1] = a[1] + 0xFFFFFFFFFFFFEULL - b[1];
    r[2] = a[2] + 0xFFFFFFFFFFFFEULL - b[2];
    r[3] = a[3] + 0xFFFFFFFFFFFFEULL - b[3];
    r[4] = a[4] + 0xFFFFFFFFFFFFEULL - b[4];
}

__host__ __device__ inline void fe_neg(fe r, const fe a)
{
    fe z;
    fe_0(z);
    fe_sub(r, z, a);
}

__host__ __device__ inline void fe_mul(fe r, const fe a, const fe b)
{
    uint64_t a1_19 = a[1] * 19, a2_19 = a[2] * 19, a3_19 = a[3] * 19, a4_19 = a[4] * 19;
    u128s t0 = { 0, 0 }, t1 = { 0, 0 }, t2 = { 0, 0 }, t3 = { 0, 0 }, t4 = { 0, 0 };

    fe_mac(&t0, a[0], b[0]); fe_mac(&t0, a1_19, b[4]); fe_mac(&t0, a2_19, b[3]); fe_mac(&t0, a3_19, b[2]); fe_mac(&t0, a4_19, b[1]);
    fe_mac(&t1, a[0], b[1]); fe_mac(&t1, a[1], b[0]); fe_mac(&t1, a2_19, b[4]); fe_mac(&t1, a3_19, b[3]); fe_mac(&t1, a4_19, b[2]);
    fe_mac(&t2, a[0], b[2]); fe_mac(&t2, a[1], b[1]); fe_mac(&t2, a[2], b[0]); fe_mac(&t2, a3_19, b[4]); fe_mac(&t2, a4_19, b[3]);
    fe_mac(&t3, a[0], b[3]); fe_mac(&t3, a[1], b[2]); fe_mac(&t3, a[2], b[1]); fe_mac(&t3, a[3], b[0]); fe_mac(&t3, a4_19, b[4]);
    fe_mac(&t4, a[0], b[4]); fe_mac(&t4, a[1], b[3]); fe_mac(&t4, a[2], b[2]); fe_mac(&t4, a[3], b[1]); fe_mac(&t4, a[4], b[0]);

    uint64_t c;
    uint64_t r0 = t0.lo & FE_M51;
    c = fe_shr51(t0);
    fe_cadd(&t1, c);
    uint64_t r1 = t1.lo & FE_M51;
    c = fe_shr51(t1);
    fe_cadd(&t2, c);
    uint64_t r2 = t2.lo & FE_M51;
    c = fe_shr51(t2);
    fe_cadd(&t3, c);
    uint64_t r3 = t3.lo & FE_M51;
    c = fe_shr51(t3);
    fe_cadd(&t4, c);
    uint64_t r4 = t4.lo & FE_M51;
    c = fe_shr51(t4);

    r0 += c * 19;
    c = r0 >> 51; r0 &= FE_M51;
    r1 += c;

    r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; r[4] = r4;
}

/* dedicated squaring: 15 wide multiplies instead of 25 */
__host__ __device__ inline void fe_sq(fe r, const fe a)
{
    uint64_t a0_2 = a[0] * 2, a1_2 = a[1] * 2, a2_2 = a[2] * 2, a3_2 = a[3] * 2;
    uint64_t a3_19 = a[3] * 19, a4_19 = a[4] * 19;
    u128s t0 = { 0, 0 }, t1 = { 0, 0 }, t2 = { 0, 0 }, t3 = { 0, 0 }, t4 = { 0, 0 };

    fe_mac(&t0, a[0], a[0]); fe_mac(&t0, a1_2, a4_19); fe_mac(&t0, a2_2, a3_19);
    fe_mac(&t1, a0_2, a[1]); fe_mac(&t1, a2_2, a4_19); fe_mac(&t1, a[3], a3_19);
    fe_mac(&t2, a0_2, a[2]); fe_mac(&t2, a[1], a[1]); fe_mac(&t2, a3_2, a4_19);
    fe_mac(&t3, a0_2, a[3]); fe_mac(&t3, a1_2, a[2]); fe_mac(&t3, a[4], a4_19);
    fe_mac(&t4, a0_2, a[4]); fe_mac(&t4, a1_2, a[3]); fe_mac(&t4, a[2], a[2]);

    uint64_t c;
    uint64_t r0 = t0.lo & FE_M51;
    c = fe_shr51(t0);
    fe_cadd(&t1, c);
    uint64_t r1 = t1.lo & FE_M51;
    c = fe_shr51(t1);
    fe_cadd(&t2, c);
    uint64_t r2 = t2.lo & FE_M51;
    c = fe_shr51(t2);
    fe_cadd(&t3, c);
    uint64_t r3 = t3.lo & FE_M51;
    c = fe_shr51(t3);
    fe_cadd(&t4, c);
    uint64_t r4 = t4.lo & FE_M51;
    c = fe_shr51(t4);

    r0 += c * 19;
    c = r0 >> 51; r0 &= FE_M51;
    r1 += c;

    r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; r[4] = r4;
}

__host__ __device__ inline uint64_t fe_load64(const uint8_t* s)
{
    uint64_t r = 0;
    for (int i = 7; i >= 0; i--)
        r = (r << 8) | s[i];
    return r;
}

__host__ __device__ inline void fe_store64(uint8_t* s, uint64_t v)
{
    for (int i = 0; i < 8; i++) {
        s[i] = (uint8_t)v;
        v >>= 8;
    }
}

/* little-endian 32 bytes -> fe (bit 255 ignored) */
__host__ __device__ inline void fe_frombytes(fe r, const uint8_t s[32])
{
    r[0] = fe_load64(s) & FE_M51;
    r[1] = (fe_load64(s + 6) >> 3) & FE_M51;
    r[2] = (fe_load64(s + 12) >> 6) & FE_M51;
    r[3] = (fe_load64(s + 19) >> 1) & FE_M51;
    r[4] = (fe_load64(s + 24) >> 12) & FE_M51;
}

/* canonical little-endian 32 bytes (fully reduced mod p) */
__host__ __device__ inline void fe_tobytes(uint8_t s[32], const fe f)
{
    fe t;
    fe_copy(t, f);
    fe_carry(t);
    fe_carry(t);

    /* now 0 <= t < 2^255; canonicalize (curve25519-donna style) */
    t[0] += 19;
    fe_carry(t);
    t[0] += 0x8000000000000ULL - 19;
    t[1] += 0x8000000000000ULL - 1;
    t[2] += 0x8000000000000ULL - 1;
    t[3] += 0x8000000000000ULL - 1;
    t[4] += 0x8000000000000ULL - 1;
    uint64_t c;
    c = t[0] >> 51; t[0] &= FE_M51; t[1] += c;
    c = t[1] >> 51; t[1] &= FE_M51; t[2] += c;
    c = t[2] >> 51; t[2] &= FE_M51; t[3] += c;
    c = t[3] >> 51; t[3] &= FE_M51; t[4] += c;
    t[4] &= FE_M51; /* drop 2^255 */

    fe_store64(s, t[0] | (t[1] << 51));
    fe_store64(s + 8, (t[1] >> 13) | (t[2] << 38));
    fe_store64(s + 16, (t[2] >> 26) | (t[3] << 25));
    fe_store64(s + 24, (t[3] >> 39) | (t[4] << 12));
}

__host__ __device__ inline int fe_isnegative(const fe f)
{
    uint8_t s[32];
    fe_tobytes(s, f);
    return s[0] & 1;
}

/* r = a^(p-2) = a^-1 mod p (ref10 addition chain) */
__host__ __device__ inline void fe_invert(fe r, const fe a)
{
    fe t0, t1, t2, t3;
    int i;

    fe_sq(t0, a);                                   /* a^2 */
    fe_sq(t1, t0);
    fe_sq(t1, t1);                                  /* a^8 */
    fe_mul(t1, a, t1);                              /* a^9 */
    fe_mul(t0, t0, t1);                             /* a^11 */
    fe_sq(t2, t0);                                  /* a^22 */
    fe_mul(t1, t1, t2);                             /* a^31 = a^(2^5-1) */
    fe_sq(t2, t1);
    for (i = 1; i < 5; i++) fe_sq(t2, t2);          /* a^(2^10-2^5) */
    fe_mul(t1, t2, t1);                             /* a^(2^10-1) */
    fe_sq(t2, t1);
    for (i = 1; i < 10; i++) fe_sq(t2, t2);
    fe_mul(t2, t2, t1);                             /* a^(2^20-1) */
    fe_sq(t3, t2);
    for (i = 1; i < 20; i++) fe_sq(t3, t3);
    fe_mul(t2, t3, t2);                             /* a^(2^40-1) */
    fe_sq(t2, t2);
    for (i = 1; i < 10; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);                             /* a^(2^50-1) */
    fe_sq(t2, t1);
    for (i = 1; i < 50; i++) fe_sq(t2, t2);
    fe_mul(t2, t2, t1);                             /* a^(2^100-1) */
    fe_sq(t3, t2);
    for (i = 1; i < 100; i++) fe_sq(t3, t3);
    fe_mul(t2, t3, t2);                             /* a^(2^200-1) */
    fe_sq(t2, t2);
    for (i = 1; i < 50; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);                             /* a^(2^250-1) */
    fe_sq(t1, t1);
    for (i = 1; i < 5; i++) fe_sq(t1, t1);
    fe_mul(r, t1, t0);                              /* a^(2^255-21) */
}
