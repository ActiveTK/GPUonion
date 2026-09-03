// Tor v3 .onion address derivation and prefix -> bitmask conversion (host only).
#pragma once
#include <stdint.h>
#include <string.h>
#include <string>
#include "keccak.h"

static const char ONION_B32[] = "abcdefghijklmnopqrstuvwxyz234567";

static inline void base32_encode(char* out, const uint8_t* in, size_t inlen)
{
    uint32_t acc = 0;
    int bits = 0;
    size_t o = 0;
    for (size_t i = 0; i < inlen; i++) {
        acc = (acc << 8) | in[i];
        bits += 8;
        while (bits >= 5) {
            out[o++] = ONION_B32[(acc >> (bits - 5)) & 31];
            bits -= 5;
        }
    }
    if (bits > 0)
        out[o++] = ONION_B32[(acc << (5 - bits)) & 31];
    out[o] = 0;
}

/* address = base32(pubkey || checksum || 0x03) + ".onion"
   checksum = SHA3-256(".onion checksum" || pubkey || 0x03)[0..1] */
static inline std::string onion_address(const uint8_t pubkey[32])
{
    uint8_t chk_in[15 + 32 + 1];
    uint8_t h[32];
    memcpy(chk_in, ".onion checksum", 15);
    memcpy(chk_in + 15, pubkey, 32);
    chk_in[47] = 0x03;
    sha3_256(h, chk_in, sizeof(chk_in));

    uint8_t data[35];
    memcpy(data, pubkey, 32);
    data[32] = h[0];
    data[33] = h[1];
    data[34] = 0x03;

    char b32[64];
    base32_encode(b32, data, 35); /* exactly 56 chars */
    return std::string(b32) + ".onion";
}

/* Build byte-level target/mask for the first bytes of the public key from a
   base32 prefix. Returns number of bytes to compare, or -1 on invalid char. */
static inline int prefix_to_mask(const char* prefix, uint8_t target[32], uint8_t mask[32])
{
    memset(target, 0, 32);
    memset(mask, 0, 32);
    size_t len = strlen(prefix);
    for (size_t j = 0; j < len; j++) {
        const char* p = strchr(ONION_B32, prefix[j]);
        if (!p)
            return -1;
        int v = (int)(p - ONION_B32);
        for (int b = 0; b < 5; b++) {
            size_t bitpos = 5 * j + b;
            size_t byte = bitpos >> 3;
            int shift = 7 - (int)(bitpos & 7);
            if (byte >= 32)
                return -1;
            target[byte] |= (uint8_t)(((v >> (4 - b)) & 1) << shift);
            mask[byte] |= (uint8_t)(1 << shift);
        }
    }
    return (int)((5 * len + 7) / 8);
}
