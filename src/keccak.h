// Minimal SHA3-256 (host only) - used for the .onion address checksum.
#pragma once
#include <stdint.h>
#include <string.h>
#include <stddef.h>

static inline uint64_t keccak_rotl64(uint64_t x, int s)
{
    return (x << s) | (x >> (64 - s));
}

static inline void keccakf1600(uint64_t st[25])
{
    static const uint64_t RC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
        0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };
    static const int rotc[24] = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
                                  27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44 };
    static const int piln[24] = { 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
                                  15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1 };

    for (int round = 0; round < 24; round++) {
        uint64_t bc[5], t;
        /* theta */
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ keccak_rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }
        /* rho + pi */
        t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = piln[i];
            bc[0] = st[j];
            st[j] = keccak_rotl64(t, rotc[i]);
            t = bc[0];
        }
        /* chi */
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (int i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        st[0] ^= RC[round];
    }
}

static inline void sha3_256(uint8_t out[32], const uint8_t* in, size_t inlen)
{
    const size_t rate = 136; /* 1088 bits */
    uint64_t st[25];
    uint8_t buf[136];
    memset(st, 0, sizeof(st));

    while (inlen >= rate) {
        for (size_t i = 0; i < rate / 8; i++) {
            uint64_t w = 0;
            for (int b = 7; b >= 0; b--)
                w = (w << 8) | in[i * 8 + b];
            st[i] ^= w;
        }
        keccakf1600(st);
        in += rate;
        inlen -= rate;
    }

    memset(buf, 0, sizeof(buf));
    memcpy(buf, in, inlen);
    buf[inlen] = 0x06;
    buf[rate - 1] |= 0x80;
    for (size_t i = 0; i < rate / 8; i++) {
        uint64_t w = 0;
        for (int b = 7; b >= 0; b--)
            w = (w << 8) | buf[i * 8 + b];
        st[i] ^= w;
    }
    keccakf1600(st);

    for (int i = 0; i < 4; i++) {
        uint64_t w = st[i];
        for (int b = 0; b < 8; b++) {
            out[i * 8 + b] = (uint8_t)w;
            w >>= 8;
        }
    }
}
