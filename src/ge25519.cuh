// ed25519 group operations, extended coordinates (X:Y:Z:T), a = -1.
// Formulas: EFD "add-2008-hwcd-3" (unified add, uses k = 2d) and "dbl-2008-hwcd".
#pragma once
#include "fe25519.cuh"

struct ge25519 {
    fe X, Y, Z, T;
};

/* k = 2d where d = -121665/121666 mod p */
__host__ __device__ inline void fe_k2d(fe r)
{
    /* d, little-endian */
    const uint8_t d_bytes[32] = {
        0xa3, 0x78, 0x59, 0x13, 0xca, 0x4d, 0xeb, 0x75,
        0xab, 0xd8, 0x41, 0x41, 0x4d, 0x0a, 0x70, 0x00,
        0x98, 0xe8, 0x79, 0x77, 0x79, 0x40, 0xc7, 0x8c,
        0x73, 0xfe, 0x6f, 0x2b, 0xee, 0x6c, 0x03, 0x52
    };
    fe_frombytes(r, d_bytes);
    fe_add(r, r, r);
}

__host__ __device__ inline void ge_identity(ge25519* r)
{
    fe_0(r->X);
    fe_1(r->Y);
    fe_1(r->Z);
    fe_0(r->T);
}

/* r = p + q (unified; handles identity and doubling inputs) */
__host__ __device__ inline void ge_add(ge25519* r, const ge25519* p, const ge25519* q, const fe k2d)
{
    fe A, B, C, D, E, F, G, H, t1, t2;

    fe_sub(t1, p->Y, p->X);
    fe_sub(t2, q->Y, q->X);
    fe_mul(A, t1, t2);
    fe_add(t1, p->Y, p->X);
    fe_add(t2, q->Y, q->X);
    fe_mul(B, t1, t2);
    fe_mul(C, p->T, q->T);
    fe_mul(C, C, k2d);
    fe_mul(D, p->Z, q->Z);
    fe_add(D, D, D);
    fe_sub(E, B, A);
    fe_sub(F, D, C);
    fe_add(G, D, C);
    fe_add(H, B, A);
    fe_mul(r->X, E, F);
    fe_mul(r->Y, G, H);
    fe_mul(r->T, E, H);
    fe_mul(r->Z, F, G);
}

/* r = 2p */
__host__ __device__ inline void ge_double(ge25519* r, const ge25519* p)
{
    fe A, B, C, D, E, F, G, H, t;

    fe_sq(A, p->X);
    fe_sq(B, p->Y);
    fe_sq(C, p->Z);
    fe_add(C, C, C);
    fe_neg(D, A);
    fe_add(t, p->X, p->Y);
    fe_sq(t, t);
    fe_sub(E, t, A);
    fe_sub(E, E, B);
    fe_add(G, D, B);
    fe_sub(F, G, C);
    fe_sub(H, D, B);
    fe_mul(r->X, E, F);
    fe_mul(r->Y, G, H);
    fe_mul(r->T, E, H);
    fe_mul(r->Z, F, G);
}

/* r = scalar * base; scalar is 32 bytes little-endian, MSB-first double-and-add.
   Not constant-time (fine for vanity search; nothing observes timing per key). */
__host__ __device__ inline void ge_scalarmult(ge25519* r, const uint8_t scalar[32], const ge25519* base, const fe k2d)
{
    ge25519 acc;
    ge_identity(&acc);
    for (int i = 255; i >= 0; i--) {
        ge_double(&acc, &acc);
        if ((scalar[i >> 3] >> (i & 7)) & 1)
            ge_add(&acc, &acc, base, k2d);
    }
    *r = acc;
}

/* standard ed25519 public-key encoding: y with sign(x) in bit 255 */
__host__ __device__ inline void ge_tobytes(uint8_t s[32], const ge25519* p)
{
    fe zi, x, y;
    fe_invert(zi, p->Z);
    fe_mul(x, p->X, zi);
    fe_mul(y, p->Y, zi);
    fe_tobytes(s, y);
    s[31] |= (uint8_t)(fe_isnegative(x) << 7);
}

/* standard base point B = (x, 4/5) */
__host__ __device__ inline void ge_basepoint(ge25519* r)
{
    const uint8_t bx[32] = {
        0x1a, 0xd5, 0x25, 0x8f, 0x60, 0x2d, 0x56, 0xc9,
        0xb2, 0xa7, 0x25, 0x95, 0x60, 0xc7, 0x2c, 0x69,
        0x5c, 0xdc, 0xd6, 0xfd, 0x31, 0xe2, 0xa4, 0xc0,
        0xfe, 0x53, 0x6e, 0xcd, 0xd3, 0x36, 0x69, 0x21
    };
    const uint8_t by[32] = {
        0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66
    };
    fe_frombytes(r->X, bx);
    fe_frombytes(r->Y, by);
    fe_1(r->Z);
    fe_mul(r->T, r->X, r->Y);
}
