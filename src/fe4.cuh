// Field arithmetic over GF(2^255-19) in 4x64-bit limbs, values kept lazily
// reduced mod 2^256-38 (= 2p) in [0, 2^256). Device path uses PTX mad/madc
// carry chains; host fallback is portable C++ (used for constants + selftest).
#pragma once
#include <stdint.h>
#include "fe25519.cuh"

typedef uint64_t fe4[4];

__host__ __device__ inline void fe4_copy(fe4 r, const fe4 a)
{
    r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; r[3] = a[3];
}

#if !defined(__CUDA_ARCH__)
/* host helper: r[idx..] += v with carry propagation (p has a guard word) */
static inline void h_accum(uint64_t* p, int idx, int len, uint64_t v)
{
    while (v && idx < len) {
        uint64_t s = p[idx] + v;
        v = s < v ? 1u : 0u;
        p[idx] = s;
        idx++;
    }
}
#endif

/* r = a * b mod 2^256-38 (lazy: result < 2^256). aliasing r==a or r==b ok. */
__host__ __device__ inline void fe4_mul(fe4 r, const fe4 a, const fe4 b)
{
#if defined(__CUDA_ARCH__)
    uint64_t p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0;

    /* p[i..i+5] += a[i]*b: one carry chain for lo halves, one for hi halves */
#define FE4_ROW(ai, o0, o1, o2, o3, o4, o5)                                   \
    asm("mad.lo.cc.u64 %0, %6, %7, %0;\n\t"                                   \
        "madc.lo.cc.u64 %1, %6, %8, %1;\n\t"                                  \
        "madc.lo.cc.u64 %2, %6, %9, %2;\n\t"                                  \
        "madc.lo.cc.u64 %3, %6, %10, %3;\n\t"                                 \
        "addc.u64 %4, %4, 0;\n\t"                                             \
        "mad.hi.cc.u64 %1, %6, %7, %1;\n\t"                                   \
        "madc.hi.cc.u64 %2, %6, %8, %2;\n\t"                                  \
        "madc.hi.cc.u64 %3, %6, %9, %3;\n\t"                                  \
        "madc.hi.cc.u64 %4, %6, %10, %4;\n\t"                                 \
        "addc.u64 %5, %5, 0;"                                                 \
        : "+l"(o0), "+l"(o1), "+l"(o2), "+l"(o3), "+l"(o4), "+l"(o5)          \
        : "l"(ai), "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]))

    FE4_ROW(a[0], p0, p1, p2, p3, p4, p5);
    FE4_ROW(a[1], p1, p2, p3, p4, p5, p6);
    FE4_ROW(a[2], p2, p3, p4, p5, p6, p7);
#undef FE4_ROW
    /* last row: carry out of p7 cannot occur (total product < 2^512) */
    asm("mad.lo.cc.u64 %0, %5, %6, %0;\n\t"
        "madc.lo.cc.u64 %1, %5, %7, %1;\n\t"
        "madc.lo.cc.u64 %2, %5, %8, %2;\n\t"
        "madc.lo.cc.u64 %3, %5, %9, %3;\n\t"
        "addc.u64 %4, %4, 0;\n\t"
        "mad.hi.cc.u64 %1, %5, %6, %1;\n\t"
        "madc.hi.cc.u64 %2, %5, %7, %2;\n\t"
        "madc.hi.cc.u64 %3, %5, %8, %3;\n\t"
        "madc.hi.u64 %4, %5, %9, %4;"
        : "+l"(p3), "+l"(p4), "+l"(p5), "+l"(p6), "+l"(p7)
        : "l"(a[3]), "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));

    /* fold: r = p[0..3] + p[4..7]*38; then fold the overflow twice more
       (after the first refold the value is < ~1500, so the last one cannot
       carry out of limb 0) */
    asm("{\n\t"
        ".reg .u64 tl0, tl1, tl2, tl3, th0, th1, th2, th3, o;\n\t"
        "mul.lo.u64 tl0, %4, 38;\n\t"
        "mul.hi.u64 th0, %4, 38;\n\t"
        "mul.lo.u64 tl1, %5, 38;\n\t"
        "mul.hi.u64 th1, %5, 38;\n\t"
        "mul.lo.u64 tl2, %6, 38;\n\t"
        "mul.hi.u64 th2, %6, 38;\n\t"
        "mul.lo.u64 tl3, %7, 38;\n\t"
        "mul.hi.u64 th3, %7, 38;\n\t"
        "add.cc.u64 %0, %0, tl0;\n\t"
        "addc.cc.u64 %1, %1, tl1;\n\t"
        "addc.cc.u64 %2, %2, tl2;\n\t"
        "addc.cc.u64 %3, %3, tl3;\n\t"
        "addc.u64 o, th3, 0;\n\t"
        "add.cc.u64 %1, %1, th0;\n\t"
        "addc.cc.u64 %2, %2, th1;\n\t"
        "addc.cc.u64 %3, %3, th2;\n\t"
        "addc.u64 o, o, 0;\n\t"
        "mad.lo.cc.u64 %0, o, 38, %0;\n\t"
        "addc.cc.u64 %1, %1, 0;\n\t"
        "addc.cc.u64 %2, %2, 0;\n\t"
        "addc.cc.u64 %3, %3, 0;\n\t"
        "addc.u64 o, 0, 0;\n\t"
        "mad.lo.u64 %0, o, 38, %0;\n\t"
        "}"
        : "+l"(p0), "+l"(p1), "+l"(p2), "+l"(p3)
        : "l"(p4), "l"(p5), "l"(p6), "l"(p7));
    r[0] = p0; r[1] = p1; r[2] = p2; r[3] = p3;
#else
    uint64_t p[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            uint64_t lo, hi;
            mul64wide(a[i], b[j], &lo, &hi);
            h_accum(p, i + j, 9, lo);
            h_accum(p, i + j + 1, 9, hi);
        }
    }
    uint64_t h[4] = { p[4], p[5], p[6], p[7] };
    uint64_t q[5] = { p[0], p[1], p[2], p[3], 0 };
    for (int j = 0; j < 4; j++) {
        uint64_t lo, hi;
        mul64wide(h[j], 38, &lo, &hi);
        h_accum(q, j, 5, lo);
        h_accum(q, j + 1, 5, hi);
    }
    while (q[4]) { /* at most twice */
        uint64_t o = q[4];
        q[4] = 0;
        h_accum(q, 0, 5, o * 38);
    }
    r[0] = q[0]; r[1] = q[1]; r[2] = q[2]; r[3] = q[3];
#endif
}

/* dedicated squaring: 6 cross products (doubled) + 4 squares = 10 wide
   multiplies instead of 16 */
__host__ __device__ inline void fe4_sq(fe4 r, const fe4 a)
{
#if defined(__CUDA_ARCH__)
    uint64_t p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0;

    /* cross products a_i*a_j (i<j) at limb offsets i+j */
    asm("mad.lo.cc.u64 %0, %5, %6, %0;\n\t"
        "madc.lo.cc.u64 %1, %5, %7, %1;\n\t"
        "madc.lo.cc.u64 %2, %5, %8, %2;\n\t"
        "addc.u64 %3, %3, 0;\n\t"
        "mad.hi.cc.u64 %1, %5, %6, %1;\n\t"
        "madc.hi.cc.u64 %2, %5, %7, %2;\n\t"
        "madc.hi.cc.u64 %3, %5, %8, %3;\n\t"
        "addc.u64 %4, %4, 0;"
        : "+l"(p1), "+l"(p2), "+l"(p3), "+l"(p4), "+l"(p5)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]));
    asm("mad.lo.cc.u64 %0, %4, %5, %0;\n\t"
        "madc.lo.cc.u64 %1, %4, %6, %1;\n\t"
        "addc.u64 %2, %2, 0;\n\t"
        "mad.hi.cc.u64 %1, %4, %5, %1;\n\t"
        "madc.hi.cc.u64 %2, %4, %6, %2;\n\t"
        "addc.u64 %3, %3, 0;"
        : "+l"(p3), "+l"(p4), "+l"(p5), "+l"(p6)
        : "l"(a[1]), "l"(a[2]), "l"(a[3]));
    asm("mad.lo.cc.u64 %0, %2, %3, %0;\n\t"
        "madc.hi.u64 %1, %2, %3, %1;"
        : "+l"(p5), "+l"(p6)
        : "l"(a[2]), "l"(a[3]));

    /* double the cross sum (p7 still zero, top carry lands there) */
    asm("add.cc.u64 %0, %0, %0;\n\t"
        "addc.cc.u64 %1, %1, %1;\n\t"
        "addc.cc.u64 %2, %2, %2;\n\t"
        "addc.cc.u64 %3, %3, %3;\n\t"
        "addc.cc.u64 %4, %4, %4;\n\t"
        "addc.cc.u64 %5, %5, %5;\n\t"
        "addc.u64 %6, 0, 0;"
        : "+l"(p1), "+l"(p2), "+l"(p3), "+l"(p4), "+l"(p5), "+l"(p6), "+l"(p7));

    /* add the squares a_i^2 at offsets 2i as one carry chain */
    asm("mad.lo.cc.u64 %0, %8, %8, %0;\n\t"
        "madc.hi.cc.u64 %1, %8, %8, %1;\n\t"
        "madc.lo.cc.u64 %2, %9, %9, %2;\n\t"
        "madc.hi.cc.u64 %3, %9, %9, %3;\n\t"
        "madc.lo.cc.u64 %4, %10, %10, %4;\n\t"
        "madc.hi.cc.u64 %5, %10, %10, %5;\n\t"
        "madc.lo.cc.u64 %6, %11, %11, %6;\n\t"
        "madc.hi.u64 %7, %11, %11, %7;"
        : "+l"(p0), "+l"(p1), "+l"(p2), "+l"(p3), "+l"(p4), "+l"(p5), "+l"(p6), "+l"(p7)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]));

    /* fold (same as fe4_mul) */
    asm("{\n\t"
        ".reg .u64 tl0, tl1, tl2, tl3, th0, th1, th2, th3, o;\n\t"
        "mul.lo.u64 tl0, %4, 38;\n\t"
        "mul.hi.u64 th0, %4, 38;\n\t"
        "mul.lo.u64 tl1, %5, 38;\n\t"
        "mul.hi.u64 th1, %5, 38;\n\t"
        "mul.lo.u64 tl2, %6, 38;\n\t"
        "mul.hi.u64 th2, %6, 38;\n\t"
        "mul.lo.u64 tl3, %7, 38;\n\t"
        "mul.hi.u64 th3, %7, 38;\n\t"
        "add.cc.u64 %0, %0, tl0;\n\t"
        "addc.cc.u64 %1, %1, tl1;\n\t"
        "addc.cc.u64 %2, %2, tl2;\n\t"
        "addc.cc.u64 %3, %3, tl3;\n\t"
        "addc.u64 o, th3, 0;\n\t"
        "add.cc.u64 %1, %1, th0;\n\t"
        "addc.cc.u64 %2, %2, th1;\n\t"
        "addc.cc.u64 %3, %3, th2;\n\t"
        "addc.u64 o, o, 0;\n\t"
        "mad.lo.cc.u64 %0, o, 38, %0;\n\t"
        "addc.cc.u64 %1, %1, 0;\n\t"
        "addc.cc.u64 %2, %2, 0;\n\t"
        "addc.cc.u64 %3, %3, 0;\n\t"
        "addc.u64 o, 0, 0;\n\t"
        "mad.lo.u64 %0, o, 38, %0;\n\t"
        "}"
        : "+l"(p0), "+l"(p1), "+l"(p2), "+l"(p3)
        : "l"(p4), "l"(p5), "l"(p6), "l"(p7));
    r[0] = p0; r[1] = p1; r[2] = p2; r[3] = p3;
#else
    fe4_mul(r, a, a);
#endif
}

/* r = a + b mod 2^256-38 (lazy) */
__host__ __device__ inline void fe4_add(fe4 r, const fe4 a, const fe4 b)
{
#if defined(__CUDA_ARCH__)
    asm("{\n\t"
        ".reg .u64 o;\n\t"
        "add.cc.u64 %0, %4, %8;\n\t"
        "addc.cc.u64 %1, %5, %9;\n\t"
        "addc.cc.u64 %2, %6, %10;\n\t"
        "addc.cc.u64 %3, %7, %11;\n\t"
        "addc.u64 o, 0, 0;\n\t"
        "mad.lo.cc.u64 %0, o, 38, %0;\n\t"
        "addc.cc.u64 %1, %1, 0;\n\t"
        "addc.cc.u64 %2, %2, 0;\n\t"
        "addc.cc.u64 %3, %3, 0;\n\t"
        "addc.u64 o, 0, 0;\n\t"
        "mad.lo.u64 %0, o, 38, %0;\n\t"
        "}"
        : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3])
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
#else
    uint64_t q[5] = { a[0], a[1], a[2], a[3], 0 };
    h_accum(q, 0, 5, b[0]);
    h_accum(q, 1, 5, b[1]);
    h_accum(q, 2, 5, b[2]);
    h_accum(q, 3, 5, b[3]);
    while (q[4]) {
        uint64_t o = q[4];
        q[4] = 0;
        h_accum(q, 0, 5, o * 38);
    }
    r[0] = q[0]; r[1] = q[1]; r[2] = q[2]; r[3] = q[3];
#endif
}

/* r = a - b mod 2^256-38 (lazy). On borrow the wrapped value (+2^256) is
   corrected by -38; a second borrow is possible only when the first-wrap
   result is < 38, and the chain then terminates. */
__host__ __device__ inline void fe4_sub(fe4 r, const fe4 a, const fe4 b)
{
#if defined(__CUDA_ARCH__)
    asm("{\n\t"
        ".reg .u64 w;\n\t"
        "sub.cc.u64 %0, %4, %8;\n\t"
        "subc.cc.u64 %1, %5, %9;\n\t"
        "subc.cc.u64 %2, %6, %10;\n\t"
        "subc.cc.u64 %3, %7, %11;\n\t"
        "subc.u64 w, 0, 0;\n\t"
        "and.b64 w, w, 38;\n\t"
        "sub.cc.u64 %0, %0, w;\n\t"
        "subc.cc.u64 %1, %1, 0;\n\t"
        "subc.cc.u64 %2, %2, 0;\n\t"
        "subc.cc.u64 %3, %3, 0;\n\t"
        "subc.u64 w, 0, 0;\n\t"
        "and.b64 w, w, 38;\n\t"
        "sub.cc.u64 %0, %0, w;\n\t"
        "subc.cc.u64 %1, %1, 0;\n\t"
        "subc.cc.u64 %2, %2, 0;\n\t"
        "subc.u64 %3, %3, 0;\n\t"
        "}"
        : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3])
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
#else
    uint64_t bw = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t d = a[i] - b[i];
        uint64_t b1 = a[i] < b[i] ? 1u : 0u;
        r[i] = d - bw;
        bw = b1 + (d < bw ? 1u : 0u);
    }
    for (int round = 0; round < 2 && bw; round++) {
        bw = 0;
        uint64_t d = r[0] - 38;
        bw = r[0] < 38 ? 1u : 0u;
        r[0] = d;
        for (int i = 1; i < 4 && bw; i++) {
            uint64_t d2 = r[i] - bw;
            bw = r[i] < bw ? 1u : 0u;
            r[i] = d2;
        }
    }
#endif
}

/* canonicalize to [0, p): conditionally subtract 2p, then p */
__host__ __device__ inline void fe4_canon(fe4 r)
{
    for (int pass = 0; pass < 2; pass++) {
        uint64_t m0 = pass == 0 ? 0xFFFFFFFFFFFFFFDAULL : 0xFFFFFFFFFFFFFFEDULL;
        uint64_t m3 = pass == 0 ? 0xFFFFFFFFFFFFFFFFULL : 0x7FFFFFFFFFFFFFFFULL;
        uint64_t t0, t1, t2, t3, bw;
#if defined(__CUDA_ARCH__)
        asm("sub.cc.u64 %0, %5, %9;\n\t"
            "subc.cc.u64 %1, %6, %10;\n\t"
            "subc.cc.u64 %2, %7, %10;\n\t"
            "subc.cc.u64 %3, %8, %11;\n\t"
            "subc.u64 %4, 0, 0;"
            : "=l"(t0), "=l"(t1), "=l"(t2), "=l"(t3), "=l"(bw)
            : "l"(r[0]), "l"(r[1]), "l"(r[2]), "l"(r[3]),
              "l"(m0), "l"(0xFFFFFFFFFFFFFFFFULL), "l"(m3));
#else
        bw = 0;
        uint64_t m[4] = { m0, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, m3 };
        uint64_t t[4];
        for (int i = 0; i < 4; i++) {
            uint64_t d = r[i] - m[i];
            uint64_t b1 = r[i] < m[i] ? 1u : 0u;
            t[i] = d - bw;
            bw = b1 + (d < bw ? 1u : 0u);
        }
        t0 = t[0]; t1 = t[1]; t2 = t[2]; t3 = t[3];
#endif
        if (!bw) {
            r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
        }
    }
}

/* r = a^-1 mod p via a^(p-2) (same ref10 chain as fe_invert); result lazy */
__host__ __device__ inline void fe4_invert(fe4 r, const fe4 a)
{
    fe4 t0, t1, t2, t3;
    int i;

    fe4_sq(t0, a);
    fe4_sq(t1, t0);
    fe4_sq(t1, t1);
    fe4_mul(t1, a, t1);
    fe4_mul(t0, t0, t1);
    fe4_sq(t2, t0);
    fe4_mul(t1, t1, t2);
    fe4_sq(t2, t1);
    for (i = 1; i < 5; i++) fe4_sq(t2, t2);
    fe4_mul(t1, t2, t1);
    fe4_sq(t2, t1);
    for (i = 1; i < 10; i++) fe4_sq(t2, t2);
    fe4_mul(t2, t2, t1);
    fe4_sq(t3, t2);
    for (i = 1; i < 20; i++) fe4_sq(t3, t3);
    fe4_mul(t2, t3, t2);
    fe4_sq(t2, t2);
    for (i = 1; i < 10; i++) fe4_sq(t2, t2);
    fe4_mul(t1, t2, t1);
    fe4_sq(t2, t1);
    for (i = 1; i < 50; i++) fe4_sq(t2, t2);
    fe4_mul(t2, t2, t1);
    fe4_sq(t3, t2);
    for (i = 1; i < 100; i++) fe4_sq(t3, t3);
    fe4_mul(t2, t3, t2);
    fe4_sq(t2, t2);
    for (i = 1; i < 50; i++) fe4_sq(t2, t2);
    fe4_mul(t1, t2, t1);
    fe4_sq(t1, t1);
    for (i = 1; i < 5; i++) fe4_sq(t1, t1);
    fe4_mul(r, t1, t0);
}

/* conversion from 5x51 (via canonical bytes) */
__host__ __device__ inline void fe4_fromfe(fe4 r, const fe f)
{
    uint8_t b[32];
    fe_tobytes(b, f);
    r[0] = fe_load64(b);
    r[1] = fe_load64(b + 8);
    r[2] = fe_load64(b + 16);
    r[3] = fe_load64(b + 24);
}

__host__ __device__ inline void fe4_tobytes(uint8_t s[32], const fe4 a)
{
    fe4 t;
    fe4_copy(t, a);
    fe4_canon(t);
    fe_store64(s, t[0]);
    fe_store64(s + 8, t[1]);
    fe_store64(s + 16, t[2]);
    fe_store64(s + 24, t[3]);
}
