// GPUonion - Tor v3 onion vanity address generator (CUDA)
//
// Algorithm (mkp224o / onion-vanity-address style, scalar-increment search):
//   base scalar a0 = random, clamped (low 3 bits = 0, bit 254 = 1, bits 253/255 = 0)
//   candidate of thread t at iteration i:  a = a0 + 8*(t + i*T)   (T = total threads)
//   public key P = a*B.  Each GPU iteration advances P by the fixed point (8T)*B,
//   so one curve addition + one field inversion per candidate.
//   Keeping increments a multiple of 8 preserves the clamped shape of the scalar,
//   so the result is a valid Tor "expanded" secret key (hs_ed25519_secret_key).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <cuda_runtime.h>

#include "fe25519.cuh"
#include "fe4.cuh"
#include "ge25519.cuh"
#include "onion.h"

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

#define CUDA_CHECK(x)                                                                 \
    do {                                                                              \
        cudaError_t err_ = (x);                                                       \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                        \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

#define MAX_RESULTS 16
#define MAX_PREFIXES 2048

/* ---------------- device constants ---------------- */
__constant__ uint8_t c_a0[32];
/* up to MAX_PREFIXES independent target/mask patterns; a candidate matches
   the search as soon as it satisfies any one of them.
   c_target/c_mask hold the full 32-byte patterns needed only by the rare
   byte-level confirm (prefix_full_match) once the hot-path bucket check
   below already thinks it has a match, so they live in plain __device__
   global memory rather than __constant__: at MAX_PREFIXES=2048 the two
   arrays alone would be 128KB, well past the 64KB constant memory budget
   that the small, every-candidate bucket-lookup tables below still need. */
__constant__ int c_nprefixes;
__device__ uint8_t c_target[MAX_PREFIXES][32];
__device__ uint8_t c_mask[MAX_PREFIXES][32];
__constant__ int c_mlen[MAX_PREFIXES];
__constant__ ge25519 c_base; /* B */
/* step point S = (8*T)*B in cached affine form for mixed addition */
__constant__ fe c_stepp;   /* Sy + Sx  */
__constant__ fe c_stepm;   /* Sy - Sx  */
__constant__ fe c_step2dt; /* 2d*Sx*Sy */
/* fast compare: prefix bits that fall into limb 0 of y (bits 0..50), one pair
   per prefix pattern */
__constant__ uint64_t c_t0[MAX_PREFIXES];
__constant__ uint64_t c_m0[MAX_PREFIXES];
/* Radix bucketing on each pattern's first base32 character (bits 3..7 of
   y's lowest byte -> 32 possible values) so a candidate only has to scan the
   handful of prefixes that could possibly start the same way, instead of all
   c_nprefixes of them every single candidate. c_bucket_start[32] holds
   prefix-sum offsets into c_bucket_idx (33 entries: 32 buckets + end
   sentinel); c_bucket_idx holds prefix indices grouped by bucket. */
__constant__ int c_bucket_start[33];
__constant__ int c_bucket_idx[MAX_PREFIXES];
/* Bloom prefilter.
   The bucket scan is fine for a handful of patterns but collapses at scale:
   every lane that reaches it walks its own bucket at its own scattered
   indices, so a single warp iteration issues up to 32 separate memory
   transactions and the scan costs the warp far more than its instruction
   count suggests. Measured on an RTX 4070, a warp that enters the scan costs
   on the order of 300 candidates' worth of elapsed time. Everything therefore
   depends on not entering it: throughput tracks 1 + ~300 * P(any lane in the
   warp survives the prefilter) across the whole range, and the pass rate is
   the only lever that matters.

   A filter is only as selective as the number of characters it keys on. One
   mask for all patterns has to be the shortest pattern's, so a single
   3-character word keys the whole filter on 15 bits: ~1800 patterns in 32768
   slots let 5% of candidates through and the filter stops working. Instead
   key at up to BLOOM_GROUPS different lengths, each pattern keyed at the
   longest of them it can fill, and probe once per length. Patterns shorter
   than the shortest key length are inserted as all of their extensions, which
   is affordable precisely because there are few of them. The host picks the
   lengths by minimizing the predicted pass rate, so a set with no short
   patterns still collapses to a single long key and a single probe. */
#define BLOOM_LOG2_BITS 17
#define BLOOM_BITS (1u << BLOOM_LOG2_BITS)
#define BLOOM_WORDS (BLOOM_BITS >> 5) /* 4096 words = 16 KB of shared memory */
#define BLOOM_GROUPS 4
#define BLOOM_HASHES 4
/* one limb-0 mask per key length in use, and how many are in use */
__constant__ uint64_t c_bloom_mask[BLOOM_GROUPS];
__constant__ int c_bloom_groups;
/* 0 for small prefix sets, where the direct bucket scan is already cheaper
   than staging the table */
__constant__ uint32_t c_use_bloom;
__device__ uint32_t g_bloom[BLOOM_WORDS];

/* ---- --use-wordlist-ab-with-normal-list: two-stage (word1 | word2) match --
   Prefixes formed by concatenating a word from list A with a word from list B
   (in either order, never A+A or B+B) are a product set, so the hot path
   never has to know about more than the first halves. c_ftinfo[p] is 0 for an
   ordinary prefix pattern - a match there is final - and otherwise packs the
   first word's character length in bits 0..4 and, in bits 5..6, which lists
   its partner may come from (bit 0 = A, bit 1 = B). A word taken from A needs
   a partner from B and vice versa, which is exactly how "no A+A, no B+B" is
   enforced; a word that appears in both lists needs a partner from either.

   Such a pattern firing only clears stage 1: the characters following the
   word must themselves be a word of length c_target_len - ftlen drawn from an
   allowed list, which stage 2 looks up in the length bucket
   [c_wl_start[l], c_wl_end[l]) of the merged word table. Every per-candidate
   cost therefore scales with |A|+|B| while the effective pattern count is
   about |A|*|B| - far past MAX_PREFIXES, and with a smaller device pattern
   count (hence a lower prefilter pass rate) than listing the pairs ever
   could. */
#define MAX_WORD_LEN 30
#define WORD_IN_A 1u
#define WORD_IN_B 2u
__constant__ int c_target_len; /* 0 when the mode is off */
/* 0 = plain prefix; else (first-word length) | (allowed partner lists << 5) */
__constant__ uint8_t c_ftinfo[MAX_PREFIXES];
__constant__ int c_wl_start[32];
__constant__ int c_wl_end[32];
/* Merged, de-duplicated A+B word table: characters as base32 values (0..31)
   grouped by length, plus which of the two lists each word belongs to. Global
   rather than __constant__ for the same reason as c_target/c_mask - only
   stage 2 reads it, and stage 2 runs at the stage-1 rate (~1e-6), not per
   candidate - and it keeps the constant budget exactly where it was. */
__device__ uint8_t g_wordchars[MAX_PREFIXES][MAX_WORD_LEN];
__device__ uint8_t g_wordmemb[MAX_PREFIXES];
/* Words of at most WORD_KEY_MAX characters also live as one packed 5-bits-per
   character integer, most significant character first, sorted ascending
   within each length bucket. That turns stage 2 from a linear walk of the
   bucket - up to a few hundred scattered byte loads, and the whole warp pays
   for one lane's - into a binary search over a contiguous array. It matters:
   once the prefilter is doing its job, nearly everything that survives is a
   genuine short-word hit, so the word lookup, not the prefilter, is what the
   surviving warps spend their time on. */
#define WORD_KEY_MAX 12 /* 12 * 5 = 60 bits */
__device__ uint64_t g_wordkey[MAX_PREFIXES];

__constant__ fe c_msp;
__constant__ fe c_msm;
__constant__ ge25519 c_stepneg;
/* 4x64 kernel: kappa = (u(S)-1)/(u(S)+1). Scaling both xADD terms by
   1/(u(S)+1) keeps (U:W) projectively identical and saves one multiply. */
__constant__ fe4 c_mk4;

struct FoundRec {
    uint32_t tid;
    uint32_t iter;
    uint32_t pfx; /* which prefix pattern (index into c_target/c_mask) matched */
    uint32_t w2;  /* wordlist mode: second word (index into g_wordchars), else ~0u */
};

/* full byte-level check against prefix pattern `p` (needed only for prefixes
   > 10 chars; also rules out the ~2^-50 limb0 false positives) */
__device__ __forceinline__ bool prefix_full_match(const uint8_t pk[32], int p)
{
    int mlen = c_mlen[p];
#pragma unroll 1
    for (int j = 0; j < mlen; j++) {
        if ((pk[j] ^ c_target[p][j]) & c_mask[p][j])
            return false;
    }
    return true;
}

/* base32 character `i` of the public key: bits 5i..5i+4 of the byte stream,
   MSB first, matching prefix_to_mask() on the host */
__device__ __forceinline__ int b32_char(const uint8_t pk[32], int i)
{
    int bit = 5 * i;
    int by = bit >> 3;
    uint32_t v = (uint32_t)pk[by] << 8;
    if (by + 1 < 32)
        v |= pk[by + 1];
    return (int)((v >> (11 - (bit & 7))) & 31);
}

/* Stage 2 of the wordlist mode. Pattern `p` matched only a first word, so the
   characters after it must form a word of the complementary length belonging
   to a list the first word is allowed to pair with; on success `w2` names it
   so the host can rebuild the full prefix. Ordinary patterns pass straight
   through. Reached only at the stage-1 rate - the sum over first words of
   32^-len, typically ~1e-6 - so the linear scan of one length bucket is far
   below the noise floor of the per-candidate curve arithmetic.

   Deliberately __noinline__ and free of any local array: inlined, its
   register demand joins the search kernels' own allocation, and at 128
   registers those sit exactly on the sm_86/sm_89 boundary where an SM fits 16
   warps. Two more registers costs a warp and ~5% throughput on every run,
   wordlist mode or not, to speed up a branch taken once in a million
   candidates. Out of line it costs nothing measurable. */
__device__ __noinline__ bool word_tail_match(const uint8_t pk[32], int p, uint32_t& w2)
{
    uint32_t info = c_ftinfo[p];
    w2 = 0xFFFFFFFFu;
    if (info == 0)
        return true;
    int ft = (int)(info & 31u);
    uint32_t need = info >> 5;
    int tl = c_target_len - ft;
    if (tl < 1 || tl > MAX_WORD_LEN)
        return false;
    int lo = c_wl_start[tl], hi = c_wl_end[tl];
    if (tl <= WORD_KEY_MAX) {
        /* pack the tail the same way the host packed the words, then binary
           search the bucket; keys are unique within a length because the word
           table is de-duplicated, so at most one word can match */
        uint64_t key = 0;
        for (int j = 0; j < tl; j++)
            key = (key << 5) | (uint64_t)b32_char(pk, ft + j);
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (g_wordkey[mid] < key)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < c_wl_end[tl] && g_wordkey[lo] == key && (g_wordmemb[lo] & need)) {
            w2 = (uint32_t)lo;
            return true;
        }
        return false;
    }
    /* longer than one packed key can hold: fall back to the byte compare */
    for (int w = lo; w < hi; w++) {
        if (!(g_wordmemb[w] & need))
            continue; /* right length, wrong list - would be an A+A or B+B pair */
        bool ok = true;
        for (int j = 0; j < tl; j++) {
            if (g_wordchars[w][j] != (uint8_t)b32_char(pk, ft + j)) {
                ok = false;
                break;
            }
        }
        if (ok) {
            w2 = (uint32_t)w;
            return true;
        }
    }
    return false;
}

/* Quick reject for the non-lazy kernels (k_search, k_search_mont): y0 is
   already the canonical low limb, so its first base32 character (bits 3..7)
   picks exactly one bucket - no offset ambiguity, unlike the fe4 lazy case
   below. Scans only that bucket instead of every configured prefix. */
__device__ __forceinline__ bool bucket_quick_any(uint64_t y0)
{
    int bk = (int)((y0 >> 3) & 0x1F);
    int end = c_bucket_start[bk + 1];
    for (int i = c_bucket_start[bk]; i < end; i++) {
        int p = c_bucket_idx[i];
        if (((y0 ^ c_t0[p]) & c_m0[p]) == 0)
            return true;
    }
    return false;
}

/* 32-bit finalizer (murmur-style). Folding the 51-bit key to 32 bits first
   keeps this to cheap 32-bit multiplies; collisions among a few thousand
   patterns in 2^32 are far below the filter's own false-positive rate. Shared
   by host (table build) and device (probe) so the two can never disagree. */
__host__ __device__ __forceinline__ uint32_t bloom_hash(uint64_t key)
{
    uint32_t x = (uint32_t)key ^ (uint32_t)(key >> 32);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/* Keys from different groups live in one table, so the group index is mixed
   in to keep two groups from aliasing on the same numeric key. */
__host__ __device__ __forceinline__ uint64_t bloom_group_key(uint64_t masked, int g)
{
    return masked + (uint64_t)g * 0x9e3779b97f4a7c15ull;
}

/* The BLOOM_HASHES bit positions a key sets or tests. Double hashing rather
   than slicing one 32-bit word: at 2^17 bits four indices need 68 bits, more
   than one hash has, and an odd stride keeps the probes distinct in a
   power-of-two table. Four, not two, because a few thousand patterns in
   128 Kbit leaves enough room that k=2 is far off the optimum - at 2000
   patterns k=2 passes 1.05e-2 of candidates and k=4 only 5.4e-4, measured. */
__host__ __device__ __forceinline__ void bloom_slots(uint64_t key, uint32_t slot[BLOOM_HASHES])
{
    uint32_t h1 = bloom_hash(key);
    uint32_t h2 = bloom_hash(key ^ 0x9e3779b97f4a7c15ull) | 1u;
    for (int i = 0; i < BLOOM_HASHES; i++)
        slot[i] = (h1 + (uint32_t)i * h2) & (BLOOM_BITS - 1);
}

__device__ __forceinline__ bool bloom_probe(const uint32_t* __restrict__ tbl, uint64_t y0)
{
    bool hit = false;
    /* c_bloom_groups is grid-uniform, so this loop is uniform too */
    for (int g = 0; g < c_bloom_groups; g++) {
        uint32_t slot[BLOOM_HASHES];
        bloom_slots(bloom_group_key(y0 & c_bloom_mask[g], g), slot);
        uint32_t acc = 1u;
        for (int i = 0; i < BLOOM_HASHES; i++)
            acc &= tbl[slot[i] >> 5] >> (slot[i] & 31);
        hit |= (acc & 1u) != 0;
    }
    return hit;
}

/* Hot-path reject: Bloom probe when the prefix set is big enough to warrant
   the table, plain bucket scan otherwise (c_use_bloom is grid-uniform, so the
   branch costs nothing).

   A survivor is far more expensive than its instruction count suggests. One
   lane getting through drags its whole warp into the bucket scan, whose
   c_bucket_idx/c_t0/c_m0 reads are scattered over tens of KB - well past the
   constant cache - and sit in a dependent loop, so the misses serialize
   instead of overlapping. Measured on an RTX 4070, a warp that enters the
   scan costs on the order of 300 candidates' worth of elapsed time, and
   throughput follows 1 + ~300 * P(any lane in the warp survives) closely
   across the whole range. That is why the filter's false-positive rate, not
   the pattern count, is what decides throughput at scale. */
__device__ __forceinline__ bool quick_any(const uint32_t* __restrict__ bloom, uint64_t y0)
{
    return c_use_bloom ? bloom_probe(bloom, y0) : bucket_quick_any(y0);
}

/* stage g_bloom into shared memory once per block; a no-op when the filter is
   off, in which case the launch requests no dynamic shared memory at all and
   occupancy is exactly what it was before */
__device__ __forceinline__ void bloom_load(uint32_t* bloom)
{
    if (c_use_bloom) {
        for (uint32_t i = threadIdx.x; i < BLOOM_WORDS; i += blockDim.x)
            bloom[i] = g_bloom[i];
    }
    __syncthreads();
}

/* thread state for Montgomery x-only stepping: projective u of P_k and P_{k-1}
   (u = (1+y)/(1-y) = (Z+Y)/(Z-Y) maps Edwards y to Curve25519 u) */
struct MontState {
    fe U, W, Um, Wm;
};

struct MontState4 {
    fe4 U, W, Um, Wm;
};

/* ---------------- kernels ---------------- */

/* each thread computes its starting point P_t = (a0 + 8t)*B */
__global__ void k_init(ge25519* pts, uint32_t nthreads)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nthreads)
        return;

    uint8_t sc[32];
    uint64_t acc = (uint64_t)t * 8ull;
    for (int i = 0; i < 32; i++) {
        acc += c_a0[i];
        sc[i] = (uint8_t)acc;
        acc >>= 8;
    }

    fe k2d;
    fe_k2d(k2d);
    ge25519 P;
    ge_scalarmult(&P, sc, &c_base, k2d);
    pts[t] = P;
}

/* same init but produces Montgomery x-only state: u(P_t) and u(P_t - S) */
__global__ void k_init_mont(MontState* st, uint32_t nthreads)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nthreads)
        return;

    uint8_t sc[32];
    uint64_t acc = (uint64_t)t * 8ull;
    for (int i = 0; i < 32; i++) {
        acc += c_a0[i];
        sc[i] = (uint8_t)acc;
        acc >>= 8;
    }

    fe k2d;
    fe_k2d(k2d);
    ge25519 P, Pm;
    ge_scalarmult(&P, sc, &c_base, k2d);
    ge_add(&Pm, &P, &c_stepneg, k2d);

    MontState s;
    fe_add(s.U, P.Z, P.Y);
    fe_sub(s.W, P.Z, P.Y);
    fe_add(s.Um, Pm.Z, Pm.Y);
    fe_sub(s.Wm, Pm.Z, Pm.Y);
    st[t] = s;
}

/* init for the 4x64 Montgomery kernel */
__global__ void k_init_mont4(MontState4* st, uint32_t nthreads)
{
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nthreads)
        return;

    uint8_t sc[32];
    uint64_t acc = (uint64_t)t * 8ull;
    for (int i = 0; i < 32; i++) {
        acc += c_a0[i];
        sc[i] = (uint8_t)acc;
        acc >>= 8;
    }

    fe k2d;
    fe_k2d(k2d);
    ge25519 P, Pm;
    ge_scalarmult(&P, sc, &c_base, k2d);
    ge_add(&Pm, &P, &c_stepneg, k2d);

    fe u, w;
    MontState4 s;
    fe_add(u, P.Z, P.Y);
    fe_sub(w, P.Z, P.Y);
    fe4_fromfe(s.U, u);
    fe4_fromfe(s.W, w);
    fe_add(u, Pm.Z, Pm.Y);
    fe_sub(w, Pm.Z, Pm.Y);
    fe4_fromfe(s.Um, u);
    fe4_fromfe(s.Wm, w);
    st[t] = s;
}

/* advance P by the cached affine step S: mixed add-2008-hwcd-3, 7 muls */
__device__ __forceinline__ void step_add(ge25519* P)
{
    fe A, B, C, D, E, F, G, H, t1;

    fe_sub(t1, P->Y, P->X);
    fe_mul(A, t1, c_stepm);
    fe_add(t1, P->Y, P->X);
    fe_mul(B, t1, c_stepp);
    fe_mul(C, P->T, c_step2dt);
    fe_add(D, P->Z, P->Z);
    fe_sub(E, B, A);
    fe_sub(F, D, C);
    fe_add(G, D, C);
    fe_add(H, B, A);
    fe_mul(P->X, E, F);
    fe_mul(P->Y, G, H);
    fe_mul(P->T, E, H);
    fe_mul(P->Z, F, G);
}

/* Batched search: each thread walks `nbatch` batches of BATCH consecutive
   candidates. Pass 1 steps the point and records (Y, Z, running product of Z);
   one field inversion then unrolls into per-candidate 1/Z (Montgomery trick).
   Matching compares limb 0 of affine y (bits 0..50) against the prefix in a
   single 64-bit op; longer prefixes fall through to a byte-level check.
   Candidate index of thread t, batch b, slot k is iter_base + b*BATCH + k. */
template <int BATCH>
__global__ void k_search(ge25519* pts, uint32_t nthreads, uint32_t nbatch,
                         uint32_t iter_base, uint32_t* found_count, FoundRec* found)
{
    extern __shared__ uint32_t s_bloom[];
    bloom_load(s_bloom);

    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nthreads)
        return;

    ge25519 P = pts[t];
    fe Ys[BATCH], Zs[BATCH], Cs[BATCH];

    for (uint32_t b = 0; b < nbatch; b++) {
#pragma unroll 1
        for (int k = 0; k < BATCH; k++) {
            fe_copy(Ys[k], P.Y);
            fe_copy(Zs[k], P.Z);
            if (k == 0)
                fe_copy(Cs[0], P.Z);
            else
                fe_mul(Cs[k], Cs[k - 1], P.Z);
            step_add(&P);
        }

        fe inv;
        fe_invert(inv, Cs[BATCH - 1]);

#pragma unroll 1
        for (int k = BATCH - 1; k >= 0; k--) {
            fe zi, y;
            if (k > 0) {
                fe_mul(zi, inv, Cs[k - 1]);
                fe_mul(inv, inv, Zs[k]);
            } else {
                fe_copy(zi, inv);
            }
            fe_mul(y, Ys[k], zi);
            uint64_t y0 = y[0];
            if (quick_any(s_bloom, y0)) {
                uint8_t pk[32];
                fe_tobytes(pk, y);
                int bk = (int)((y0 >> 3) & 0x1F);
                int end = c_bucket_start[bk + 1];
                for (int i = c_bucket_start[bk]; i < end; i++) {
                    int p = c_bucket_idx[i];
                    uint32_t w2 = 0xFFFFFFFFu;
                    if (((y0 ^ c_t0[p]) & c_m0[p]) == 0 && prefix_full_match(pk, p) &&
                        word_tail_match(pk, p, w2)) {
                        uint32_t idx = atomicAdd(found_count, 1);
                        if (idx < MAX_RESULTS) {
                            found[idx].tid = t;
                            found[idx].iter = iter_base + b * BATCH + k;
                            found[idx].pfx = p;
                            found[idx].w2 = w2;
                        }
                    }
                }
            }
        }
    }
    pts[t] = P;
}

/* Montgomery x-only batched search. Candidate y = (U-W)/(U+W); stepping is a
   differential addition (xADD) with the fixed affine step S: 4 muls + 2 squares
   per candidate instead of 7 muls for the Edwards mixed add. The batched
   inversion runs over D = U+W. */
template <int BATCH>
__global__ void k_search_mont(MontState* st, uint32_t nthreads, uint32_t nbatch,
                              uint32_t iter_base, uint32_t* found_count, FoundRec* found)
{
    extern __shared__ uint32_t s_bloom[];
    bloom_load(s_bloom);

    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nthreads)
        return;

    MontState S = st[t];
    fe Ns[BATCH], Ds[BATCH], Cs[BATCH];

    for (uint32_t b = 0; b < nbatch; b++) {
#pragma unroll 1
        for (int k = 0; k < BATCH; k++) {
            fe_sub_nc(Ns[k], S.U, S.W);
            fe_add_nc(Ds[k], S.U, S.W);
            if (k == 0)
                fe_copy(Cs[0], Ds[0]);
            else
                fe_mul(Cs[k], Cs[k - 1], Ds[k]);

            /* xADD: P_{k+1} = P_k + S given P_{k-1} = P_k - S */
            fe t1, t2, a, d, Un;
            fe_mul(t1, Ns[k], c_msp); /* (U-W)(uS+1) */
            fe_mul(t2, Ds[k], c_msm); /* (U+W)(uS-1) */
            fe_add_nc(a, t1, t2);
            fe_sub_nc(d, t1, t2);
            fe_sq(a, a);
            fe_sq(d, d);
            fe_mul(Un, S.Wm, a);
            fe_mul(d, S.Um, d);
            fe_copy(S.Um, S.U);
            fe_copy(S.Wm, S.W);
            fe_copy(S.U, Un);
            fe_copy(S.W, d);
        }

        fe inv;
        fe_invert(inv, Cs[BATCH - 1]);

#pragma unroll 1
        for (int k = BATCH - 1; k >= 0; k--) {
            fe zi, y;
            if (k > 0) {
                fe_mul(zi, inv, Cs[k - 1]);
                fe_mul(inv, inv, Ds[k]);
            } else {
                fe_copy(zi, inv);
            }
            fe_mul(y, Ns[k], zi);
            uint64_t y0 = y[0];
            if (quick_any(s_bloom, y0)) {
                uint8_t pk[32];
                fe_tobytes(pk, y);
                int bk = (int)((y0 >> 3) & 0x1F);
                int end = c_bucket_start[bk + 1];
                for (int i = c_bucket_start[bk]; i < end; i++) {
                    int p = c_bucket_idx[i];
                    uint32_t w2 = 0xFFFFFFFFu;
                    if (((y0 ^ c_t0[p]) & c_m0[p]) == 0 && prefix_full_match(pk, p) &&
                        word_tail_match(pk, p, w2)) {
                        uint32_t idx = atomicAdd(found_count, 1);
                        if (idx < MAX_RESULTS) {
                            found[idx].tid = t;
                            found[idx].iter = iter_base + b * BATCH + k;
                            found[idx].pfx = p;
                            found[idx].w2 = w2;
                        }
                    }
                }
            }
        }
    }
    st[t] = S;
}

/* 4x64 Montgomery x-only batched search: same algorithm as k_search_mont but
   with lazy mod 2^256-38 arithmetic and PTX madc carry chains. */
template <int BATCH>
__global__ void k_search_mont4(MontState4* __restrict__ st, uint32_t nthreads, uint32_t nbatch,
                               uint32_t iter_base, uint32_t* __restrict__ found_count,
                               FoundRec* __restrict__ found)
{
    extern __shared__ uint32_t s_bloom[];
    bloom_load(s_bloom);

    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nthreads)
        return;

    MontState4 S = st[t];
    /* Ms[k] = N_k * C_{k-1} (prefix product of D folded in during pass 1),
       so the backward pass needs only y = Ms[k] * inv with inv = (C_k)^-1
       and the running-inverse update inv *= Ds[k]: two arrays, not three. */
    fe4 Ms[BATCH], Ds[BATCH];

    for (uint32_t b = 0; b < nbatch; b++) {
        fe4 C;
#pragma unroll 1
        for (int k = 0; k < BATCH; k++) {
            fe4 N;
            fe4_sub(N, S.U, S.W);
            fe4_add(Ds[k], S.U, S.W);
            if (k == 0) {
                fe4_copy(Ms[0], N);
                fe4_copy(C, Ds[0]);
            } else {
                fe4_mul(Ms[k], N, C);
                fe4_mul(C, C, Ds[k]);
            }

            /* xADD scaled by 1/(u(S)+1): t2 = D*kappa, a = N+t2, d = N-t2 */
            fe4 t2, a, d, Un;
            fe4_mul(t2, Ds[k], c_mk4);
            fe4_add(a, N, t2);
            fe4_sub(d, N, t2);
            fe4_sq(a, a);
            fe4_sq(d, d);
            fe4_mul(Un, S.Wm, a);
            fe4_mul(d, S.Um, d);
            fe4_copy(S.Um, S.U);
            fe4_copy(S.Wm, S.W);
            fe4_copy(S.U, Un);
            fe4_copy(S.W, d);
        }

        fe4 inv;
        fe4_invert(inv, C);

#pragma unroll 1
        for (int k = BATCH - 1; k >= 0; k--) {
            fe4 y;
            fe4_mul(y, Ms[k], inv);
            if (k > 0)
                fe4_mul(inv, inv, Ds[k]);
            /* Canonicalize before probing rather than after. y is lazy - its
               value is y_true + j*p for j in {0,1,2}, so limb 0 is y[0] + 19j
               - and a filter keyed on limb 0 would otherwise have to probe all
               three offsets. That costs three times the probe work, and, worse,
               three times the survivor rate: the two bogus offsets are
               effectively random 51-bit values, so they clear the filter
               structurally exactly as often as the real one does, and a
               survivor drags its whole warp into the pattern scan. fe4_canon
               is ~26 instructions against ~1700 for the candidate itself, so
               it pays for itself twice over. */
            fe4_canon(y);
            uint64_t yc = y[0];
            if (quick_any(s_bloom, yc)) {
                uint8_t pk[32];
                fe4_tobytes(pk, y);
                int bk = (int)((yc >> 3) & 0x1F);
                int end = c_bucket_start[bk + 1];
                for (int i = c_bucket_start[bk]; i < end; i++) {
                    int p = c_bucket_idx[i];
                    uint32_t w2 = 0xFFFFFFFFu;
                    if (((yc ^ c_t0[p]) & c_m0[p]) == 0 && prefix_full_match(pk, p) &&
                        word_tail_match(pk, p, w2)) {
                        uint32_t idx = atomicAdd(found_count, 1);
                        if (idx < MAX_RESULTS) {
                            found[idx].tid = t;
                            found[idx].iter = iter_base + b * BATCH + k;
                            found[idx].pfx = p;
                            found[idx].w2 = w2;
                        }
                    }
                }
            }
        }
    }
    st[t] = S;
}

/* ---------------- host helpers ---------------- */

static void os_random(uint8_t* buf, size_t len)
{
#ifdef _WIN32
    if (BCryptGenRandom(NULL, buf, (ULONG)len, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
        fprintf(stderr, "BCryptGenRandom failed\n");
        exit(1);
    }
#else
    FILE* f = fopen("/dev/urandom", "rb");
    if (!f || fread(buf, 1, len, f) != len) {
        fprintf(stderr, "failed to read /dev/urandom\n");
        exit(1);
    }
    fclose(f);
#endif
}

static std::string hex(const uint8_t* b, size_t n)
{
    static const char* H = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += H[b[i] >> 4];
        s += H[b[i] & 15];
    }
    return s;
}

/* Reads prefixes from a file, one per line (blank lines skipped, surrounding
   whitespace and CR/LF trimmed). Each line is validated immediately so a bad
   entry is reported with its file + line number rather than surfacing later
   mixed in with prefixes from other sources. Appends onto `out` (lower-
   cased). Returns false (after printing an error) if the file can't be
   opened or any non-blank line is not a valid prefix. */
static bool load_prefixes_from_file(const std::string& path, std::vector<std::string>& out)
{
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot open prefix file '%s'\n", path.c_str());
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        lineno++;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos)
            continue; /* blank line */
        line = line.substr(start);
        for (auto& ch : line)
            if (ch >= 'A' && ch <= 'Z')
                ch += 32;
        uint8_t target[32], mask[32];
        int mlen = prefix_to_mask(line.c_str(), target, mask);
        if (mlen < 0 || line.size() > 30) {
            fprintf(stderr, "invalid prefix '%s' at %s:%d (allowed: a-z 2-7, max 30 chars)\n",
                    line.c_str(), path.c_str(), lineno);
            return false;
        }
        out.push_back(line);
    }
    return true;
}

/* Reads a word list. Words may be one per line or comma-separated, in any
   mix, so a generator can emit either shape; blank entries are skipped. Each
   word is validated on the spot with the same rules as a prefix, so a bad
   entry is reported with its file:line before any GPU work starts. */
static bool load_words_from_file(const std::string& path, std::vector<std::string>& out)
{
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot open word list '%s'\n", path.c_str());
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        lineno++;
        size_t pos = 0;
        while (pos <= line.size()) {
            size_t comma = line.find(',', pos);
            std::string tok =
                line.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            pos = comma == std::string::npos ? line.size() + 1 : comma + 1;

            while (!tok.empty() && (tok.back() == '\r' || tok.back() == ' ' || tok.back() == '\t'))
                tok.pop_back();
            size_t start = tok.find_first_not_of(" \t");
            if (start == std::string::npos)
                continue; /* blank line or empty field */
            tok = tok.substr(start);
            for (auto& ch : tok)
                if (ch >= 'A' && ch <= 'Z')
                    ch += 32;
            uint8_t target[32], mask[32];
            if (prefix_to_mask(tok.c_str(), target, mask) < 0 || tok.size() > MAX_WORD_LEN) {
                fprintf(stderr, "invalid word '%s' at %s:%d (allowed: a-z 2-7, max %d chars)\n",
                        tok.c_str(), path.c_str(), lineno, MAX_WORD_LEN);
                return false;
            }
            out.push_back(tok);
        }
    }
    return true;
}

/* human-readable key rate: switches unit prefix so the number stays 3-4 digits */
static std::string format_rate(double keys_per_sec)
{
    char buf[64];
    if (keys_per_sec >= 1e9)
        snprintf(buf, sizeof(buf), "%.2f GKey/s", keys_per_sec / 1e9);
    else if (keys_per_sec >= 1e6)
        snprintf(buf, sizeof(buf), "%.2f MKey/s", keys_per_sec / 1e6);
    else
        snprintf(buf, sizeof(buf), "%.2f kKey/s", keys_per_sec / 1e3);
    return buf;
}

/* scalar = a0 + 8*(t + i*T), little-endian 32 bytes */
static void result_scalar(uint8_t out[32], const uint8_t a0[32], uint32_t t, uint64_t i, uint64_t T)
{
    uint64_t acc = 8ull * ((uint64_t)t + i * T);
    for (int j = 0; j < 32; j++) {
        acc += a0[j];
        out[j] = (uint8_t)acc;
        acc >>= 8;
    }
}

static void host_pubkey(uint8_t pk[32], const uint8_t scalar[32])
{
    fe k2d;
    fe_k2d(k2d);
    ge25519 B, P;
    ge_basepoint(&B);
    ge_scalarmult(&P, scalar, &B, k2d);
    ge_tobytes(pk, &P);
}

static bool write_file(const std::filesystem::path& p, const void* data, size_t len)
{
    std::ofstream f(p, std::ios::binary);
    if (!f)
        return false;
    f.write((const char*)data, (std::streamsize)len);
    return f.good();
}

static bool save_result(const std::string& outdir, const std::string& addr,
                        const uint8_t scalar[32], const uint8_t pubkey[32])
{
    namespace fs = std::filesystem;
    fs::path dir = fs::path(outdir) / addr.substr(0, addr.find('.'));
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        fprintf(stderr, "failed to create %s\n", dir.string().c_str());
        return false;
    }

    /* hs_ed25519_secret_key: header(32) + expanded key (scalar || random nonce-key) */
    uint8_t sec[32 + 64];
    memset(sec, 0, sizeof(sec));
    memcpy(sec, "== ed25519v1-secret: type0 ==", 29);
    memcpy(sec + 32, scalar, 32);
    os_random(sec + 64, 32);

    uint8_t pub[32 + 32];
    memset(pub, 0, sizeof(pub));
    memcpy(pub, "== ed25519v1-public: type0 ==", 29);
    memcpy(pub + 32, pubkey, 32);

    std::string hostname = addr + "\n";
    fs::path secpath = dir / "hs_ed25519_secret_key";
    bool ok = write_file(secpath, sec, sizeof(sec)) &&
              write_file(dir / "hs_ed25519_public_key", pub, sizeof(pub)) &&
              write_file(dir / "hostname", hostname.data(), hostname.size());
    if (ok)
        printf("  saved to %s\n", dir.string().c_str());
    return ok;
}

/* POST `local_path` to end2end.tech under `remote_name`. Shared by upload_key
   (a found secret key) and upload_status_log (a periodic progress snapshot)
   so a result survives the instance being stopped. On failure the file still
   exists on local disk. Requires curl on PATH (present by default on
   Windows 10+ and typical Linux images). */
/* --no-upload. Off by default and never set implicitly: a key that silently
   fails to leave the machine is a lost key and a wasted search, so the only
   way to stop an upload is to ask for it on the command line, and every
   skipped upload still says so and names the local file. */
static bool g_no_upload = false;

static void curl_upload(const std::filesystem::path& local_path, const std::string& remote_name)
{
    if (g_no_upload) {
        printf("  upload skipped (--no-upload), kept locally: %s\n",
               local_path.string().c_str());
        return;
    }
    /* -F value quoted so paths with spaces survive; remote_name is either
       base32 + ".key"/".onion" or a generated log filename, so it needs no
       escaping itself */
    std::string cmd = "curl -s --max-time 60 -X POST https://api.end2end.tech/upload "
                      "-F \"file=@" + local_path.string() + ";filename=" + remote_name + "\"";

#ifdef _WIN32
    FILE* p = _popen(cmd.c_str(), "r");
#else
    FILE* p = popen(cmd.c_str(), "r");
#endif
    if (!p) {
        fprintf(stderr, "  upload: failed to run curl (kept locally: %s)\n",
                local_path.string().c_str());
        return;
    }
    std::string resp;
    char chunk[512];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), p)) > 0)
        resp.append(chunk, n);
#ifdef _WIN32
    int rc = _pclose(p);
#else
    int rc = pclose(p);
#endif
    if (rc != 0 || resp.find("\"Status\"") == std::string::npos) {
        fprintf(stderr, "  upload FAILED (kept locally: %s): %s\n", local_path.string().c_str(),
                resp.empty() ? "no response" : resp.c_str());
        return;
    }
    /* pull out the download URL for convenience */
    size_t u = resp.find("\"URL\"");
    if (u != std::string::npos) {
        size_t s = resp.find('"', resp.find(':', u) + 1);
        size_t e = s == std::string::npos ? s : resp.find('"', s + 1);
        if (e != std::string::npos) {
            std::string url = resp.substr(s + 1, e - s - 1);
            size_t pos;
            while ((pos = url.find("\\/")) != std::string::npos) /* unescape JSON */
                url.erase(pos, 1);
            printf("  uploaded: %s\n", url.c_str());
            return;
        }
    }
    printf("  uploaded %s\n", remote_name.c_str());
}

static void upload_key(const std::string& outdir, const std::string& addr)
{
    namespace fs = std::filesystem;
    fs::path secpath = fs::path(outdir) / addr.substr(0, addr.find('.')) / "hs_ed25519_secret_key";
    curl_upload(secpath, addr + ".key");
}

/* Renders a plain-text progress snapshot: elapsed time, combined key rate,
   matches found so far, the configured prefixes, and every address found so
   far. Used by --upload-status-per-30min so progress isn't lost if the
   instance is stopped before a match is found. */
static std::string build_status_text(double elapsed, uint64_t total_keys, int found, int want,
                                     bool keep_forever, size_t ndevices,
                                     const std::vector<std::string>& prefixes,
                                     const std::vector<std::string>& found_addrs)
{
    char buf[256];
    std::string s = "GPUonion status\n";
    double rate = elapsed > 0.0 ? (double)total_keys / elapsed : 0.0;
    snprintf(buf, sizeof(buf), "elapsed  : %.0f s\ntotal keys: %llu\nrate     : %s\ndevices  : %zu\n",
             elapsed, (unsigned long long)total_keys, format_rate(rate).c_str(), ndevices);
    s += buf;
    if (keep_forever)
        snprintf(buf, sizeof(buf), "matches  : %d (--keep-working-until-ctrlc)\n", found);
    else
        snprintf(buf, sizeof(buf), "matches  : %d / %d\n", found, want);
    s += buf;
    snprintf(buf, sizeof(buf), "prefixes (%zu):\n", prefixes.size());
    s += buf;
    for (const auto& p : prefixes)
        s += "  " + p + "\n";
    s += "found addresses:\n";
    for (const auto& a : found_addrs)
        s += "  " + a + "\n";
    return s;
}

/* Writes `content` to <outdir>/status.log and uploads it as `remote_name`. */
static void upload_status_log(const std::string& outdir, const std::string& remote_name,
                              const std::string& content)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(outdir, ec);
    fs::path logpath = fs::path(outdir) / "status.log";
    if (!write_file(logpath, content.data(), content.size())) {
        fprintf(stderr, "  status log: failed to write %s\n", logpath.string().c_str());
        return;
    }
    curl_upload(logpath, remote_name);
}

/* ---------------- selftest ---------------- */

static bool selftest()
{
    fe k2d;
    fe_k2d(k2d);
    ge25519 B;
    ge_basepoint(&B);

    /* 1. encoding of 1*B */
    uint8_t sc[32], pk[32];
    memset(sc, 0, 32);
    sc[0] = 1;
    ge25519 P;
    ge_scalarmult(&P, sc, &B, k2d);
    ge_tobytes(pk, &P);
    uint8_t expect[32];
    memset(expect, 0x66, 32);
    expect[0] = 0x58;
    if (memcmp(pk, expect, 32) != 0) {
        fprintf(stderr, "selftest FAIL: basepoint encoding: %s\n", hex(pk, 32).c_str());
        return false;
    }

    /* 2. double(B) == B + B */
    ge25519 D, S;
    uint8_t d1[32], d2[32];
    ge_double(&D, &B);
    ge_add(&S, &B, &B, k2d);
    ge_tobytes(d1, &D);
    ge_tobytes(d2, &S);
    if (memcmp(d1, d2, 32) != 0) {
        fprintf(stderr, "selftest FAIL: double vs add\n");
        return false;
    }

    /* 3. (a+b)*B == a*B + b*B */
    uint8_t sa[32], sb[32], sab[32];
    memset(sa, 0, 32); memset(sb, 0, 32);
    sa[0] = 0x15; sa[3] = 0x37; sa[17] = 0x9c;
    sb[0] = 0xf1; sb[8] = 0x42; sb[30] = 0x1d;
    uint32_t carry = 0;
    for (int i = 0; i < 32; i++) {
        uint32_t v = (uint32_t)sa[i] + sb[i] + carry;
        sab[i] = (uint8_t)v;
        carry = v >> 8;
    }
    ge25519 PA, PB, PS1, PS2;
    ge_scalarmult(&PA, sa, &B, k2d);
    ge_scalarmult(&PB, sb, &B, k2d);
    ge_scalarmult(&PS1, sab, &B, k2d);
    ge_add(&PS2, &PA, &PB, k2d);
    uint8_t e1[32], e2[32];
    ge_tobytes(e1, &PS1);
    ge_tobytes(e2, &PS2);
    if (memcmp(e1, e2, 32) != 0) {
        fprintf(stderr, "selftest FAIL: scalar homomorphism\n");
        return false;
    }

    /* 3b. fe_sq consistency with fe_mul */
    {
        fe x, s1, s2;
        fe_copy(x, B.X);
        fe_mul(x, x, B.Y);
        fe_sq(s1, x);
        fe_mul(s2, x, x);
        uint8_t b1[32], b2[32];
        fe_tobytes(b1, s1);
        fe_tobytes(b2, s2);
        if (memcmp(b1, b2, 32) != 0) {
            fprintf(stderr, "selftest FAIL: fe_sq vs fe_mul\n");
            return false;
        }
    }

    /* 3c. fe4 (4x64 lazy) arithmetic vs fe (5x51) */
    {
        fe x, yv, tmp;
        fe_copy(x, B.X);
        fe_copy(yv, B.Y);
        fe4 x4, y4, r4;
        fe4_fromfe(x4, x);
        fe4_fromfe(y4, yv);
        uint8_t b1[32], b2[32];
        const char* names[4] = { "mul", "add", "sub", "invert" };
        for (int op = 0; op < 4; op++) {
            switch (op) {
            case 0: fe_mul(tmp, x, yv); fe4_mul(r4, x4, y4); break;
            case 1: fe_add(tmp, x, yv); fe4_add(r4, x4, y4); break;
            case 2: fe_sub(tmp, x, yv); fe4_sub(r4, x4, y4); break;
            case 3: fe_invert(tmp, x); fe4_invert(r4, x4); break;
            }
            fe_tobytes(b1, tmp);
            fe4_tobytes(b2, r4);
            if (memcmp(b1, b2, 32) != 0) {
                fprintf(stderr, "selftest FAIL: fe4 %s\n", names[op]);
                return false;
            }
        }
    }

    /* 4. field canonicalization: p -> 0, p+1 -> 1 */
    uint8_t pb[32];
    memset(pb, 0xff, 32);
    pb[0] = 0xed;
    pb[31] = 0x7f;
    fe f;
    uint8_t fb[32], zero32[32];
    memset(zero32, 0, 32);
    fe_frombytes(f, pb);
    fe_tobytes(fb, f);
    if (memcmp(fb, zero32, 32) != 0) {
        fprintf(stderr, "selftest FAIL: tobytes(p) != 0\n");
        return false;
    }
    pb[0] = 0xee;
    fe_frombytes(f, pb);
    fe_tobytes(fb, f);
    zero32[0] = 1;
    if (memcmp(fb, zero32, 32) != 0) {
        fprintf(stderr, "selftest FAIL: tobytes(p+1) != 1\n");
        return false;
    }

    /* 5. known onion address derivation sanity: 56 chars + .onion */
    std::string addr = onion_address(pk);
    if (addr.size() != 62 || addr.substr(56) != ".onion") {
        fprintf(stderr, "selftest FAIL: address format %s\n", addr.c_str());
        return false;
    }
    return true;
}

/* ---------------- per-device worker ---------------- */

/* Everything one GPU needs to run fully independently: its own random start
   scalar, its own device buffers, its own progress/found output. Several of
   these run concurrently, one host thread per selected GPU (CUDA contexts
   are per-thread, so cudaSetDevice() here scopes all subsequent CUDA calls
   on this thread to `device` without any cross-thread locking). `want` is a
   shared target across all workers via `global_found`; `out_mtx` only
   serializes stdout/stderr writes so lines from different GPUs don't
   interleave mid-line. */
struct RunConfig {
    std::vector<std::string> prefixes;
    uint8_t target[MAX_PREFIXES][32];
    uint8_t mask[MAX_PREFIXES][32];
    int mlen[MAX_PREFIXES];
    int nprefixes;
    int tpb_arg;
    int blocks_arg;
    uint32_t iters;
    int batch;
    int mode; /* 0 = fe4 Montgomery, 1 = 5x51 Montgomery, 2 = Edwards ext */
    std::string outdir;
    int want;
    bool benchmark;
    double bench_secs;
    bool keep_forever; /* --keep-working-until-ctrlc: never stop on match count */
    bool upload_status; /* --upload-status-per-30min: periodic progress snapshot */
    /* --use-wordlist-ab-with-normal-list. target_len == 0 means the mode is
       off, in which case ftinfo is all zero and the device skips stage 2. */
    int target_len;
    uint8_t ftinfo[MAX_PREFIXES];    /* 0 = plain prefix, else len | (partner lists << 5) */
    std::vector<std::string> words;  /* merged A+B, ordered by character length */
    std::vector<uint8_t> wordchars;  /* the same, base32 values, MAX_WORD_LEN per word */
    std::vector<uint8_t> wordmemb;   /* per word: WORD_IN_A | WORD_IN_B */
    std::vector<uint64_t> wordkey;   /* the same, packed 5 bits per character */
    int wl_start[32];
    int wl_end[32];
};

/* threads-per-block / blocks-per-grid a device will actually launch with,
   given the user's -t/--blocks overrides (0 = unset -> default). Shared by
   the startup banner (main) and the worker (run_device) so they never
   disagree about how many threads a GPU is running. */
static void resolve_launch(const RunConfig& cfg, const cudaDeviceProp& prop, int& tpb, int& blocks)
{
    tpb = cfg.tpb_arg;
    blocks = cfg.blocks_arg;
    if (tpb <= 0)
        tpb = 128;
    if (blocks <= 0)
        blocks = prop.multiProcessorCount * 16;
}

/* What the prefilter will be: which key lengths, the masks that select them,
   the populated table, and the pass rate the choice is predicted to give.
   Built by plan_bloom() so the startup banner and every worker agree. */
struct BloomPlan {
    int ngroups = 0;
    int key_len[BLOOM_GROUPS] = {0};
    uint64_t mask[BLOOM_GROUPS] = {0};
    double pass = 0.0;      /* predicted fraction of candidates that survive */
    double insertions = 0.0;/* entries in the table, expansions included */
    uint32_t use_bloom = 0;
    std::vector<uint32_t> table;
};

/* limb-0 (bits 0..50) view of a byte-level target or mask */
static uint64_t limb0_of(const uint8_t b[32])
{
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--)
        v = (v << 8) | b[i];
    return v & FE_M51;
}

/* Chooses the prefilter's key lengths and builds its table. Keying every
   pattern at the shortest one's length - the only option with a single mask -
   means one short word destroys the filter for everything else, so instead
   try every set of up to BLOOM_GROUPS key lengths and keep the one with the
   lowest predicted pass rate. A pattern is keyed at the longest chosen length
   it can fill; one shorter than every chosen length is inserted as all of its
   extensions, which is affordable exactly because such patterns are rare.
   With no short patterns the search naturally lands on a single long key and
   a single probe, so the common case pays nothing for the machinery. */
static void plan_bloom(const RunConfig& cfg, const uint64_t* t0, BloomPlan& out)
{
    const int KEY_LEN_MAX = 10;        /* 10 chars = 50 bits, all limb 0 holds */
    const double INSERT_BUDGET = 60000.0;

    /* limb-0 mask of an L-character prefix, and the limb-0 bits a character
       value contributes at a given position - enough to build any extension
       key without going back through the byte-level encoder */
    uint64_t len_mask[KEY_LEN_MAX + 1];
    static uint64_t charbit[KEY_LEN_MAX][32];
    {
        uint8_t tt[32], mm[32];
        len_mask[0] = 0;
        for (int L = 1; L <= KEY_LEN_MAX; L++) {
            std::string dummy(L, 'a');
            prefix_to_mask(dummy.c_str(), tt, mm);
            len_mask[L] = limb0_of(mm);
        }
        for (int pos = 0; pos < KEY_LEN_MAX; pos++)
            for (int v = 0; v < 32; v++) {
                std::string dummy(pos, 'a');
                dummy += ONION_B32[v];
                prefix_to_mask(dummy.c_str(), tt, mm);
                charbit[pos][v] = limb0_of(tt);
            }
    }

    std::vector<int> plen(cfg.nprefixes);
    for (int p = 0; p < cfg.nprefixes; p++)
        plen[p] = (int)cfg.prefixes[p].size();

    double best_pass = 1e300;
    int best[BLOOM_GROUPS] = {0}, best_n = 0;
    double best_ins = 0.0;
    for (int a = 1; a <= KEY_LEN_MAX; a++)
        for (int b = a; b <= KEY_LEN_MAX; b++)
            for (int c = b; c <= KEY_LEN_MAX; c++)
            for (int d = c; d <= KEY_LEN_MAX; d++) {
                int uniq[BLOOM_GROUPS];
                int u = 0;
                uniq[u++] = a;
                if (b > a)
                    uniq[u++] = b;
                if (c > b)
                    uniq[u++] = c;
                if (d > c)
                    uniq[u++] = d;

                double ins_g[BLOOM_GROUPS] = {0.0};
                double total = 0.0;
                for (int q = 0; q < cfg.nprefixes; q++) {
                    int gi = 0;
                    for (int i = 0; i < u; i++)
                        if (uniq[i] <= plen[q])
                            gi = i;
                    double mult = plen[q] < uniq[gi] ? pow(32.0, uniq[gi] - plen[q]) : 1.0;
                    ins_g[gi] += mult;
                    total += mult;
                    if (total > INSERT_BUDGET)
                        break;
                }
                if (total > INSERT_BUDGET)
                    continue;

                double structural = 0.0;
                for (int i = 0; i < u; i++)
                    structural += ins_g[i] * pow(32.0, -(double)uniq[i]);
                double load = (double)BLOOM_HASHES * total / (double)BLOOM_BITS;
                double fp = pow(1.0 - exp(-load), (double)BLOOM_HASHES);
                /* one probe per group per candidate, on the canonical value */
                double pass = structural + (double)u * fp;
                if (pass < best_pass) {
                    best_pass = pass;
                    best_n = u;
                    best_ins = total;
                    for (int i = 0; i < u; i++)
                        best[i] = uniq[i];
                }
            }

    out.ngroups = best_n;
    out.pass = best_pass;
    out.insertions = best_ins;
    for (int i = 0; i < best_n; i++) {
        out.key_len[i] = best[i];
        out.mask[i] = len_mask[best[i]];
    }
    out.use_bloom = (cfg.nprefixes > 8 && best_n > 0) ? 1u : 0u;
    out.table.assign(BLOOM_WORDS, 0u);
    if (!out.use_bloom)
        return;

    for (int q = 0; q < cfg.nprefixes; q++) {
        int gi = 0;
        for (int i = 0; i < best_n; i++)
            if (best[i] <= plen[q])
                gi = i;
        int L = best[gi];
        auto insert = [&](uint64_t key) {
            uint32_t slot[BLOOM_HASHES];
            bloom_slots(bloom_group_key(key & out.mask[gi], gi), slot);
            for (int r = 0; r < BLOOM_HASHES; r++)
                out.table[slot[r] >> 5] |= 1u << (slot[r] & 31);
        };
        if (plen[q] >= L) {
            insert(t0[q]);
            continue;
        }
        /* shorter than its key length: insert every extension, so the filter
           still keys on L characters for everything else */
        int freec = L - plen[q];
        uint64_t base = t0[q] & len_mask[plen[q]];
        std::vector<int> odo(freec, 0);
        for (;;) {
            uint64_t key = base;
            for (int j = 0; j < freec; j++)
                key |= charbit[plen[q] + j][odo[j]];
            insert(key);
            int j = freec - 1;
            while (j >= 0 && ++odo[j] == 32)
                odo[j--] = 0;
            if (j < 0)
                break;
        }
    }
}

/* benchmark outcome for one GPU, filled in just before the worker returns;
   the caller aggregates these across all devices once every thread has
   joined, so the final speed/expected-time summary reflects combined
   throughput instead of one block per GPU */
struct BenchResult {
    uint64_t keys = 0;
    double elapsed = 0.0;
};

static void run_device(int device, const RunConfig& cfg, std::atomic<int>& global_found,
                       std::atomic<uint64_t>& g_total_keys, std::atomic<int>& g_crosscheck_ok,
                       std::chrono::steady_clock::time_point t_start, std::mutex& out_mtx,
                       std::vector<std::string>& g_found_addrs, BenchResult* bench_out)
{
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int tpb, blocks;
    resolve_launch(cfg, prop, tpb, blocks);
    uint32_t T = (uint32_t)blocks * (uint32_t)tpb;
    uint32_t nbatch = (cfg.iters + (uint32_t)cfg.batch - 1) / (uint32_t)cfg.batch;
    if (nbatch == 0)
        nbatch = 1;
    uint32_t launch_iters = nbatch * (uint32_t)cfg.batch;
    int batch = cfg.batch;
    int mode = cfg.mode;

    /* base scalar: random per device, clamped, bit 253 cleared for increment headroom */
    uint8_t a0[32];
    os_random(a0, 32);
    a0[0] &= 0xF8;
    a0[31] = (uint8_t)((a0[31] & 0x1F) | 0x40);

    /* host-side constants */
    fe k2d;
    fe_k2d(k2d);
    ge25519 hostB, step;
    ge_basepoint(&hostB);
    uint8_t step_sc[32];
    memset(step_sc, 0, 32);
    uint64_t sv = 8ull * T;
    for (int i = 0; i < 8; i++)
        step_sc[i] = (uint8_t)(sv >> (8 * i));
    ge_scalarmult(&step, step_sc, &hostB, k2d);

    /* normalize step to affine and cache (Sy+Sx, Sy-Sx, 2d*Sx*Sy) for mixed add */
    fe szi, sx, sy, st, stepp, stepm, step2dt;
    fe_invert(szi, step.Z);
    fe_mul(sx, step.X, szi);
    fe_mul(sy, step.Y, szi);
    fe_add(stepp, sy, sx);
    fe_sub(stepm, sy, sx);
    fe_mul(st, sx, sy);
    fe_mul(step2dt, st, k2d);

    /* Montgomery step constants: u(S) = (1+Sy)/(1-Sy), cached as u(S)+1, u(S)-1;
       and -S in Edwards extended coordinates for computing P_t - S at init */
    fe one, us, tmp, msp, msm;
    fe_1(one);
    fe_add(tmp, one, sy);
    fe_sub(us, one, sy);
    fe_invert(us, us);
    fe_mul(us, us, tmp);
    fe_add(msp, us, one);
    fe_sub(msm, us, one);
    ge25519 stepneg;
    fe_neg(stepneg.X, step.X);
    fe_copy(stepneg.Y, step.Y);
    fe_copy(stepneg.Z, step.Z);
    fe_neg(stepneg.T, step.T);

    /* prefix bits that live in limb 0 of y (bits 0..50) for the fast compare,
       one pair per prefix pattern */
    uint64_t m0[MAX_PREFIXES], t0[MAX_PREFIXES];
    for (int p = 0; p < cfg.nprefixes; p++) {
        uint64_t mm = 0, tt = 0;
        for (int i = 7; i >= 0; i--) {
            mm = (mm << 8) | cfg.mask[p][i];
            tt = (tt << 8) | cfg.target[p][i];
        }
        m0[p] = mm & FE_M51;
        t0[p] = tt & FE_M51;
    }

    /* group prefix indices by their first base32 character (top 5 bits of
       target byte 0, always fully masked since every prefix is >= 1 char) so
       the search kernels only scan a bucket instead of every prefix */
    int bucket_start[33] = {0};
    int bucket_idx[MAX_PREFIXES];
    for (int p = 0; p < cfg.nprefixes; p++)
        bucket_start[((cfg.target[p][0] >> 3) & 0x1F) + 1]++;
    for (int b = 0; b < 32; b++)
        bucket_start[b + 1] += bucket_start[b];
    {
        int cursor[32];
        memcpy(cursor, bucket_start, sizeof(cursor));
        for (int p = 0; p < cfg.nprefixes; p++)
            bucket_idx[cursor[(cfg.target[p][0] >> 3) & 0x1F]++] = p;
    }

    /* prefilter: key lengths and table, chosen the same way for every device */
    BloomPlan plan;
    plan_bloom(cfg, t0, plan);
    const size_t search_shmem = plan.use_bloom ? sizeof(uint32_t) * BLOOM_WORDS : 0;

    CUDA_CHECK(cudaMemcpyToSymbol(c_bloom_mask, plan.mask, sizeof(plan.mask)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bloom_groups, &plan.ngroups, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_use_bloom, &plan.use_bloom, sizeof(uint32_t)));
    if (plan.use_bloom)
        CUDA_CHECK(
            cudaMemcpyToSymbol(g_bloom, plan.table.data(), sizeof(uint32_t) * BLOOM_WORDS));
    CUDA_CHECK(cudaMemcpyToSymbol(c_a0, a0, 32));
    CUDA_CHECK(cudaMemcpyToSymbol(c_nprefixes, &cfg.nprefixes, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_target, cfg.target, sizeof(cfg.target)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, cfg.mask, sizeof(cfg.mask)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_mlen, cfg.mlen, sizeof(cfg.mlen)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_target_len, &cfg.target_len, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ftinfo, cfg.ftinfo, sizeof(cfg.ftinfo)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_wl_start, cfg.wl_start, sizeof(cfg.wl_start)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_wl_end, cfg.wl_end, sizeof(cfg.wl_end)));
    if (!cfg.wordchars.empty()) {
        CUDA_CHECK(cudaMemcpyToSymbol(g_wordchars, cfg.wordchars.data(), cfg.wordchars.size()));
        CUDA_CHECK(cudaMemcpyToSymbol(g_wordmemb, cfg.wordmemb.data(), cfg.wordmemb.size()));
        CUDA_CHECK(cudaMemcpyToSymbol(g_wordkey, cfg.wordkey.data(),
                                      sizeof(uint64_t) * cfg.wordkey.size()));
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_base, &hostB, sizeof(ge25519)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_stepp, stepp, sizeof(fe)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_stepm, stepm, sizeof(fe)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_step2dt, step2dt, sizeof(fe)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_t0, t0, sizeof(uint64_t) * cfg.nprefixes));
    CUDA_CHECK(cudaMemcpyToSymbol(c_m0, m0, sizeof(uint64_t) * cfg.nprefixes));
    CUDA_CHECK(cudaMemcpyToSymbol(c_bucket_start, bucket_start, sizeof(bucket_start)));
    if (cfg.nprefixes > 0)
        CUDA_CHECK(cudaMemcpyToSymbol(c_bucket_idx, bucket_idx, sizeof(int) * cfg.nprefixes));
    CUDA_CHECK(cudaMemcpyToSymbol(c_msp, msp, sizeof(fe)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_msm, msm, sizeof(fe)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_stepneg, &stepneg, sizeof(ge25519)));
    fe kappa;
    fe_invert(kappa, msp);
    fe_mul(kappa, kappa, msm);
    fe4 mk4;
    fe4_fromfe(mk4, kappa);
    CUDA_CHECK(cudaMemcpyToSymbol(c_mk4, mk4, sizeof(fe4)));

    static_assert(sizeof(MontState) == sizeof(ge25519), "state buffers share allocation");
    static_assert(sizeof(MontState4) <= sizeof(ge25519), "state buffers share allocation");
    void* d_state;
    uint32_t* d_count;
    FoundRec* d_found;
    CUDA_CHECK(cudaMalloc(&d_state, sizeof(ge25519) * T));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(FoundRec) * MAX_RESULTS));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint32_t)));

    if (mode == 2)
        k_init<<<blocks, tpb>>>((ge25519*)d_state, T);
    else if (mode == 1)
        k_init_mont<<<blocks, tpb>>>((MontState*)d_state, T);
    else
        k_init_mont4<<<blocks, tpb>>>((MontState4*)d_state, T);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* cross-check GPU init against host computation for a few threads */
    for (uint32_t t = 0; t < 3; t++) {
        uint8_t sc[32], hpk[32], gy[32];
        result_scalar(sc, a0, t, 0, T);
        host_pubkey(hpk, sc);
        if (mode == 2) {
            ge25519 chk;
            CUDA_CHECK(cudaMemcpy(&chk, (ge25519*)d_state + t, sizeof(chk),
                                  cudaMemcpyDeviceToHost));
            ge_tobytes(gy, &chk);
        } else if (mode == 1) {
            MontState chk{};
            CUDA_CHECK(cudaMemcpy(&chk, (MontState*)d_state + t, sizeof(chk),
                                  cudaMemcpyDeviceToHost));
            fe n, dd, y;
            fe_sub(n, chk.U, chk.W);
            fe_add(dd, chk.U, chk.W);
            fe_invert(dd, dd);
            fe_mul(y, n, dd);
            fe_tobytes(gy, y);
            hpk[31] &= 0x7f; /* Montgomery state has no x sign */
        } else {
            MontState4 chk;
            CUDA_CHECK(cudaMemcpy(&chk, (MontState4*)d_state + t, sizeof(chk),
                                  cudaMemcpyDeviceToHost));
            fe4 n, dd, y;
            fe4_sub(n, chk.U, chk.W);
            fe4_add(dd, chk.U, chk.W);
            fe4_invert(dd, dd);
            fe4_mul(y, n, dd);
            fe4_tobytes(gy, y);
            hpk[31] &= 0x7f;
        }
        if (memcmp(hpk, gy, 32) != 0) {
            std::lock_guard<std::mutex> lk(out_mtx);
            fprintf(stderr, "device %d: GPU/CPU mismatch at init - aborting\n", device);
            cudaFree(d_state);
            cudaFree(d_count);
            cudaFree(d_found);
            return;
        }
    }
    g_crosscheck_ok.fetch_add(1, std::memory_order_relaxed);

    uint64_t total_keys = 0;
    uint64_t iter_base = 0;

    auto launch_search = [&](uint32_t ib) {
        if (mode == 0) {
            MontState4* p = (MontState4*)d_state;
            switch (batch) {
            case 8:  k_search_mont4<8><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 16: k_search_mont4<16><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 32: k_search_mont4<32><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 64: k_search_mont4<64><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 128: k_search_mont4<128><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 256: k_search_mont4<256><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 512: k_search_mont4<512><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 1024: k_search_mont4<1024><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            }
        } else if (mode == 2) {
            ge25519* p = (ge25519*)d_state;
            switch (batch) {
            case 8:  k_search<8><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 16: k_search<16><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 32: k_search<32><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 64: k_search<64><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 128: k_search<128><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 256: k_search<256><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 512: k_search<512><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            }
        } else {
            MontState* p = (MontState*)d_state;
            switch (batch) {
            case 8:  k_search_mont<8><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 16: k_search_mont<16><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 32: k_search_mont<32><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 64: k_search_mont<64><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 128: k_search_mont<128><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 256: k_search_mont<256><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 512: k_search_mont<512><<<blocks, tpb, search_shmem>>>(p, T, nbatch, ib, d_count, d_found); break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    if (cfg.benchmark) {
        /* one warmup launch, excluded from timing */
        launch_search(0);
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint32_t)));
        iter_base = launch_iters;
    }

    while (cfg.benchmark || cfg.keep_forever || global_found.load() < cfg.want) {
        if (iter_base + launch_iters > 0xFFFFFFFFull) {
            std::lock_guard<std::mutex> lk(out_mtx);
            fprintf(stderr, "device %d: iteration counter exhausted; restart with a new seed\n",
                    device);
            break;
        }
        launch_search((uint32_t)iter_base);
        iter_base += launch_iters;
        total_keys += (uint64_t)T * launch_iters;
        g_total_keys.fetch_add((uint64_t)T * launch_iters, std::memory_order_relaxed);

        uint32_t n = 0;
        CUDA_CHECK(cudaMemcpy(&n, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (n > 0) {
            FoundRec recs[MAX_RESULTS];
            uint32_t nn = n > MAX_RESULTS ? MAX_RESULTS : n;
            CUDA_CHECK(cudaMemcpy(recs, d_found, sizeof(FoundRec) * nn, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint32_t)));

            for (uint32_t r = 0; r < nn && (cfg.keep_forever || global_found.load() < cfg.want); r++) {
                uint8_t sc[32], pk[32];
                result_scalar(sc, a0, recs[r].tid, recs[r].iter, T);
                host_pubkey(pk, sc);
                std::string addr = onion_address(pk);
                /* rebuild the full pattern: in wordlist mode cfg.prefixes[]
                   holds only the first word and the device reports which word
                   completed it, so the host still verifies the whole thing */
                std::string matched;
                if (recs[r].pfx < cfg.prefixes.size()) {
                    matched = cfg.prefixes[recs[r].pfx];
                    if (recs[r].w2 < cfg.words.size())
                        matched += cfg.words[recs[r].w2];
                }
                if (matched.empty() || addr.compare(0, matched.size(), matched) != 0) {
                    std::lock_guard<std::mutex> lk(out_mtx);
                    fprintf(stderr, "WARNING: candidate failed host verification (%s), skipped\n",
                            addr.c_str());
                    continue;
                }
                double el = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t_start).count();
                {
                    std::lock_guard<std::mutex> lk(out_mtx);
                    printf("\nFOUND (%.1fs, %llu keys total): %s\n", el,
                           (unsigned long long)g_total_keys.load(), addr.c_str());
                    printf("  secret scalar: %s\n", hex(sc, 32).c_str());
                    if (save_result(cfg.outdir, addr, sc, pk))
                        upload_key(cfg.outdir, addr);
                    g_found_addrs.push_back(addr);
                }
                global_found.fetch_add(1);
            }
        }

        double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_start).count();
        if (cfg.benchmark && el >= cfg.bench_secs) {
            if (bench_out) {
                bench_out->keys = total_keys;
                bench_out->elapsed = el;
            }
            break;
        }
    }

    cudaFree(d_state);
    cudaFree(d_count);
    cudaFree(d_found);
}

/* ---------------- main ---------------- */

static void usage(const char* argv0)
{
    fprintf(stderr,
            "GPUonion - Tor v3 onion vanity address generator (CUDA)\n"
            "usage: %s <prefix> [<prefix> ...] [options]\n"
            "       %s -b [options]\n"
            "  <prefix>       base32 prefix(es) to search for (chars a-z 2-7).\n"
            "                 pass more than one (as separate args and/or comma-\n"
            "                 separated) to match any of them; the first hit wins.\n"
            "                 up to %d prefixes, max 30 chars each.\n"
            "  --prefix-from-file <path>\n"
            "                 also read prefixes from a file, one per line (blank\n"
            "                 lines skipped). Combined with any given on the command\n"
            "                 line. Every line is validated up front; an invalid one\n"
            "                 aborts with its file:line before any GPU work starts.\n"
            "  --use-wordlist-ab-with-normal-list <list-A> <list-B> <len> <normal-list>\n"
            "                 build prefixes by concatenating one word from A\n"
            "                 with one word from B, in either order, so the\n"
            "                 result is exactly <len> chars. A+A and B+B are\n"
            "                 never paired, so \"200 nouns x 600 adjectives\"\n"
            "                 stays readable. Word lists take one word per line\n"
            "                 and/or comma-separated words. The pairs are never\n"
            "                 expanded: the GPU matches the first word, then\n"
            "                 checks the tail against the words of the\n"
            "                 complementary length, so the effective pattern\n"
            "                 count is a product and can reach millions while\n"
            "                 the device still sees only |A|+|B| patterns.\n"
            "                 <normal-list> is read exactly like --prefix-from-file\n"
            "                 and searched alongside. Constraint: the merged word\n"
            "                 list plus the normal list is at most %d entries.\n"
            "  -b, --bench    run a ~20 second benchmark (no prefix needed)\n"
            "  -d <spec>      CUDA device(s): \"all\" for every visible GPU (default),\n"
            "                 a single index, or a comma list e.g. \"0,1,2\".\n"
            "                 Each GPU runs fully independently in its own host\n"
            "                 thread (own random start point, own output).\n"
            "  -n <count>     stop after this many matches, summed across all\n"
            "                 selected GPUs (default 1) - with multiple prefixes\n"
            "                 the default already stops at the first hit, no\n"
            "                 matter which pattern in the list it matched\n"
            "  --keep-working-until-ctrlc\n"
            "                 ignore -n and keep searching (saving every match)\n"
            "                 until interrupted (Ctrl+C)\n"
            "  --upload-status-per-30min\n"
            "                 every 30 minutes, upload a progress snapshot\n"
            "                 (elapsed time, rate, matches found so far) so\n"
            "                 progress survives the instance being stopped\n"
            "  -o <dir>       output directory (default ./found)\n"
            "  --no-upload    do not upload found keys or status snapshots; keep\n"
            "                 them in the output directory only. Every skipped\n"
            "                 upload is reported, so a run that keeps its keys\n"
            "                 local is never confused with one that uploaded.\n"
            "  --blocks <n>   blocks per launch (default: SMs * 8)\n"
            "  -t <threads>   threads per block (default 256)\n"
            "  -i <iters>     candidates per thread per launch (default 512)\n"
            "  -B <batch>     Montgomery inversion batch size: 8..512 (default 128)\n"
            "  --m51          use the 5x51-limb Montgomery kernel (for A/B)\n"
            "  --ext          use Edwards extended-coordinate stepping (slower; for A/B)\n"
            "  --selftest     run internal tests only\n",
            argv0,
            argv0,
            MAX_PREFIXES,
            MAX_PREFIXES);
}

int main(int argc, char** argv)
{
    std::vector<std::string> prefix_args; /* raw positional args, may still hold comma lists */
    std::string prefix_file; /* --prefix-from-file: one prefix per line */
    /* --use-wordlist-ab-with-normal-list <A> <B> <target len> <normal list> */
    std::string wl_path_a, wl_path_b, wl_normal_path;
    int wl_target_len = 0;
    std::string device_arg = "all"; /* single index, comma list ("0,1"), or "all" (default) */
    int tpb = 0, blocks = 0, batch = 512;
    uint32_t iters = 1024;
    int want = 1;
    int mode = 0; /* 0 = fe4 Montgomery, 1 = 5x51 Montgomery, 2 = Edwards ext */
    std::string outdir = "found";
    bool selftest_only = false;
    bool benchmark = false;
    bool keep_forever = false;
    bool upload_status = false;
    const double bench_secs = 20.0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--selftest")) selftest_only = true;
        else if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "--bench")) benchmark = true;
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) device_arg = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) want = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--keep-working-until-ctrlc")) keep_forever = true;
        else if (!strcmp(argv[i], "--upload-status-per-30min")) upload_status = true;
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--no-upload")) g_no_upload = true;
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) tpb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-B") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--m51")) mode = 1;
        else if (!strcmp(argv[i], "--ext")) mode = 2;
        else if (!strcmp(argv[i], "--prefix-from-file") && i + 1 < argc) prefix_file = argv[++i];
        else if (!strcmp(argv[i], "--use-wordlist-ab-with-normal-list") && i + 4 < argc) {
            wl_path_a = argv[++i];
            wl_path_b = argv[++i];
            wl_target_len = atoi(argv[++i]);
            wl_normal_path = argv[++i];
        }
        else if (argv[i][0] != '-') prefix_args.push_back(argv[i]);
        else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!selftest()) {
        fprintf(stderr, "internal selftest failed - refusing to run\n");
        return 1;
    }
    if (selftest_only) {
        printf("selftest OK\n");
        return 0;
    }
    if (prefix_args.empty() && prefix_file.empty() && wl_path_a.empty() && !benchmark) {
        usage(argv[0]);
        return 1;
    }

    /* split each positional arg on ',' so "-d foo,bar" style comma lists
       work alongside plain separate arguments */
    std::vector<std::string> raw_prefixes;
    for (const auto& a : prefix_args) {
        size_t pos = 0;
        while (pos <= a.size()) {
            size_t comma = a.find(',', pos);
            std::string tok = a.substr(pos, comma == std::string::npos ? std::string::npos
                                                                        : comma - pos);
            if (!tok.empty())
                raw_prefixes.push_back(tok);
            if (comma == std::string::npos)
                break;
            pos = comma + 1;
        }
    }

    /* --prefix-from-file adds to (not replaces) anything given on the
       command line; each line is validated as it's read, so a bad entry is
       reported with its file:line before any GPU work starts */
    if (!prefix_file.empty() && !load_prefixes_from_file(prefix_file, raw_prefixes))
        return 1;

    /* --use-wordlist-ab-with-normal-list. The normal list is read exactly like
       --prefix-from-file. The pairs are NOT expanded: every word of A and of B
       that can start a target_len pair becomes one device pattern, carrying
       the list(s) its partner may come from, and the tail is checked on the
       GPU by word_tail_match. |A|+|B| device patterns therefore stand for
       about |A|*|B|*2 real prefixes. A word with no legal partner of the
       complementary length is dropped - keeping it would only buy stage-1
       hits that can never complete. */
    std::vector<uint8_t> raw_ftinfo;   /* per raw_prefixes entry: 0, or len | (need << 5) */
    std::vector<std::string> wl_words; /* merged A+B, ordered by character length */
    std::vector<uint8_t> wl_memb;      /* per wl_words entry: WORD_IN_A | WORD_IN_B */
    int wl_cnt[32] = {0};              /* words per character length */
    int wl_cnt_memb[32][4] = {{0}};    /* words per length matching a partner mask */
    if (!wl_path_a.empty()) {
        if (wl_target_len < 2 || wl_target_len > MAX_WORD_LEN) {
            fprintf(stderr, "target length must be 2..%d (got %d)\n", MAX_WORD_LEN,
                    wl_target_len);
            return 1;
        }
        std::vector<std::string> words_a, words_b, normal_raw;
        if (!load_words_from_file(wl_path_a, words_a))
            return 1;
        if (!load_words_from_file(wl_path_b, words_b))
            return 1;
        if (!load_prefixes_from_file(wl_normal_path, normal_raw))
            return 1;

        /* merge into one table, remembering which list(s) each word came from:
           a word in both A and B may pair with either side, one in A only must
           pair with B, and vice versa - that is what rules out A+A and B+B */
        {
            std::map<std::string, uint8_t> memb;
            for (const auto& w : words_a)
                memb[w] |= (uint8_t)WORD_IN_A;
            for (const auto& w : words_b)
                memb[w] |= (uint8_t)WORD_IN_B;
            if (memb.size() + normal_raw.size() > (size_t)MAX_PREFIXES) {
                fprintf(stderr,
                        "%zu distinct words + %zu normal-list entries = %zu, over the %d limit\n",
                        memb.size(), normal_raw.size(), memb.size() + normal_raw.size(),
                        MAX_PREFIXES);
                return 1;
            }
            /* order by length so every length is one contiguous index range
               that stage 2 can scan directly */
            std::vector<std::pair<size_t, std::string>> bylen;
            for (const auto& kv : memb)
                bylen.push_back({kv.first.size(), kv.first});
            /* by length first, then lexicographically - which for base32 is
               the same order as the packed keys stage 2 binary searches */
            std::stable_sort(bylen.begin(), bylen.end(),
                             [](const std::pair<size_t, std::string>& x,
                                const std::pair<size_t, std::string>& y) {
                                 if (x.first != y.first)
                                     return x.first < y.first;
                                 return x.second < y.second;
                             });
            for (const auto& e : bylen) {
                wl_words.push_back(e.second);
                wl_memb.push_back(memb[e.second]);
            }
        }
        for (size_t i = 0; i < wl_words.size(); i++) {
            int L = (int)wl_words[i].size();
            wl_cnt[L]++;
            for (int need = 1; need <= 3; need++)
                if (wl_memb[i] & need)
                    wl_cnt_memb[L][need]++;
        }

        for (const auto& nl : normal_raw)
            raw_prefixes.push_back(nl);
        raw_ftinfo.assign(raw_prefixes.size(), 0);

        int nfirst = 0;
        for (size_t i = 0; i < wl_words.size(); i++) {
            /* a word from A needs a partner from B and vice versa */
            uint32_t need = ((wl_memb[i] & WORD_IN_A) ? WORD_IN_B : 0u) |
                            ((wl_memb[i] & WORD_IN_B) ? WORD_IN_A : 0u);
            int comp = wl_target_len - (int)wl_words[i].size();
            if (comp < 1 || comp > MAX_WORD_LEN || wl_cnt_memb[comp][need] == 0)
                continue; /* no legal partner of the complementary length */
            raw_prefixes.push_back(wl_words[i]);
            raw_ftinfo.push_back((uint8_t)(wl_words[i].size() | (need << 5)));
            nfirst++;
        }
        if (nfirst == 0) {
            fprintf(stderr, "no word of '%s' pairs with a word of '%s' to make %d characters\n",
                    wl_path_a.c_str(), wl_path_b.c_str(), wl_target_len);
            return 1;
        }
    }

    /* benchmark: search a 16-char prefix (expected 32^16 keys - will never
       match) so the per-candidate work is identical to a real search */
    if (benchmark && raw_prefixes.empty())
        raw_prefixes.push_back("bench234bench234");

    if (raw_prefixes.empty() && !benchmark) {
        fprintf(stderr, "no prefixes given (check --prefix-from-file contents)\n");
        return 1;
    }

    if ((int)raw_prefixes.size() > MAX_PREFIXES) {
        fprintf(stderr, "too many prefixes (%zu, max %d)\n", raw_prefixes.size(), MAX_PREFIXES);
        return 1;
    }

    /* normalize + validate each prefix */
    raw_ftinfo.resize(raw_prefixes.size(), 0);
    RunConfig cfg{};
    cfg.target_len = wl_path_a.empty() ? 0 : wl_target_len;
    cfg.words = wl_words;
    cfg.wordmemb = wl_memb;
    {
        int acc = 0;
        for (int L = 0; L < 32; L++) {
            cfg.wl_start[L] = acc;
            acc += wl_cnt[L];
            cfg.wl_end[L] = acc;
        }
        cfg.wordchars.assign(wl_words.size() * MAX_WORD_LEN, 0);
        cfg.wordkey.assign(wl_words.size(), 0);
        for (size_t i = 0; i < wl_words.size(); i++) {
            uint64_t key = 0;
            for (size_t j = 0; j < wl_words[i].size(); j++) {
                uint8_t v = (uint8_t)(strchr(ONION_B32, wl_words[i][j]) - ONION_B32);
                cfg.wordchars[i * MAX_WORD_LEN + j] = v;
                if (j < WORD_KEY_MAX)
                    key = (key << 5) | v;
            }
            cfg.wordkey[i] = key;
        }
    }
    cfg.nprefixes = (int)raw_prefixes.size();
    for (int p = 0; p < cfg.nprefixes; p++) {
        std::string pfx = raw_prefixes[p];
        for (auto& ch : pfx)
            if (ch >= 'A' && ch <= 'Z')
                ch += 32;
        uint8_t target[32], mask[32];
        int mlen = prefix_to_mask(pfx.c_str(), target, mask);
        if (mlen < 0 || pfx.empty() || pfx.size() > 30) {
            fprintf(stderr, "invalid prefix '%s' (allowed: a-z 2-7, max 30 chars)\n",
                    pfx.c_str());
            return 1;
        }
        memcpy(cfg.target[p], target, 32);
        memcpy(cfg.mask[p], mask, 32);
        cfg.mlen[p] = mlen;
        cfg.ftinfo[p] = raw_ftinfo[p];
        cfg.prefixes.push_back(pfx);
    }

    if (batch != 8 && batch != 16 && batch != 32 && batch != 64 && batch != 128 &&
        batch != 256 && batch != 512 && batch != 1024) {
        fprintf(stderr, "batch size must be a power of two in 8..1024\n");
        return 1;
    }
    if (batch == 1024 && mode != 0) {
        fprintf(stderr, "batch 1024 is only supported by the default (fe4) kernel\n");
        return 1;
    }

    /* resolve device_arg into a concrete list: "all" -> every visible GPU,
       "a,b,c" -> that list, otherwise a single index (default "0") */
    std::vector<int> devices;
    if (device_arg == "all") {
        int cnt = 0;
        CUDA_CHECK(cudaGetDeviceCount(&cnt));
        if (cnt <= 0) {
            fprintf(stderr, "no CUDA devices found\n");
            return 1;
        }
        for (int i = 0; i < cnt; i++)
            devices.push_back(i);
    } else {
        size_t pos = 0;
        while (pos <= device_arg.size()) {
            size_t comma = device_arg.find(',', pos);
            std::string tok = device_arg.substr(pos, comma == std::string::npos
                                                          ? std::string::npos
                                                          : comma - pos);
            if (!tok.empty())
                devices.push_back(atoi(tok.c_str()));
            if (comma == std::string::npos)
                break;
            pos = comma + 1;
        }
        if (devices.empty()) {
            fprintf(stderr, "invalid -d value '%s'\n", device_arg.c_str());
            return 1;
        }
    }

    cfg.tpb_arg = tpb;
    cfg.blocks_arg = blocks;
    cfg.iters = iters;
    cfg.batch = batch;
    cfg.mode = mode;
    cfg.outdir = outdir;
    cfg.want = want;
    cfg.benchmark = benchmark;
    cfg.bench_secs = bench_secs;
    cfg.keep_forever = keep_forever;
    cfg.upload_status = upload_status;

    /* Combined match probability across all patterns (treated as mutually
       exclusive, which holds unless patterns overlap each other). A first-word
       pattern is not a match on its own: it stands for every target_len pair
       starting with that word, so it contributes (number of words of the
       complementary length) * 32^-target_len. stage1_sum is the separate rate
       at which a pattern fires and pulls its whole warp into the second stage
       - worth reporting, because short first words make that the dominant cost
       long before the pattern count does. */
    double prob_sum = 0.0, stage1_sum = 0.0;
    uint64_t effective_patterns = 0;
    int shortest_len = MAX_WORD_LEN + 1, nfirst_words = 0;
    for (int p = 0; p < cfg.nprefixes; p++) {
        int plen = (int)cfg.prefixes[p].size();
        stage1_sum += pow(32.0, -(double)plen);
        if (plen < shortest_len)
            shortest_len = plen;
        if (cfg.ftinfo[p] == 0) {
            prob_sum += pow(32.0, -(double)plen);
            effective_patterns += 1;
        } else {
            int comp = cfg.target_len - (int)(cfg.ftinfo[p] & 31u);
            uint64_t n2 = (uint64_t)wl_cnt_memb[comp][cfg.ftinfo[p] >> 5];
            prob_sum += (double)n2 * pow(32.0, -(double)cfg.target_len);
            effective_patterns += n2;
            nfirst_words++;
        }
    }
    double expected = prob_sum > 0.0 ? 1.0 / prob_sum : 0.0;

    /* What the prefilter will do with this pattern set. Worth printing for
       every mode, not just the wordlist one: the survivor rate is what
       decides throughput, and it is set by the pattern lengths rather than
       by anything the user can see directly. */
    BloomPlan banner_plan;
    {
        std::vector<uint64_t> t0h(cfg.nprefixes);
        for (int q = 0; q < cfg.nprefixes; q++)
            t0h[q] = limb0_of(cfg.target[q]);
        plan_bloom(cfg, t0h.data(), banner_plan);
    }

    /* combined startup banner: group devices with identical name/SM count so
       a large fleet prints one line, not one block of text per GPU */
    {
        struct DevGroup { std::string name; int major, minor, sms, count; };
        std::vector<DevGroup> groups;
        uint64_t T_total = 0;
        for (int d : devices) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
            int tpb_r, blocks_r;
            resolve_launch(cfg, prop, tpb_r, blocks_r);
            T_total += (uint64_t)tpb_r * (uint64_t)blocks_r;
            bool merged = false;
            for (auto& g : groups) {
                if (g.name == prop.name && g.major == prop.major && g.minor == prop.minor &&
                    g.sms == prop.multiProcessorCount) {
                    g.count++;
                    merged = true;
                    break;
                }
            }
            if (!merged)
                groups.push_back({prop.name, prop.major, prop.minor, prop.multiProcessorCount, 1});
        }

        printf("devices : %zu GPU%s - ", devices.size(), devices.size() == 1 ? "" : "s");
        for (size_t i = 0; i < groups.size(); i++) {
            const auto& g = groups[i];
            printf("%s%dx %s (sm_%d%d, %d SMs)", i ? ", " : "", g.count, g.name.c_str(), g.major,
                   g.minor, g.sms);
        }
        printf("\n");

        if (benchmark) {
            printf("mode    : benchmark (~%.0f s)\n", bench_secs);
        } else if (cfg.target_len > 0) {
            printf("wordlist: %zu distinct A+B words, %d usable as the first half"
                   " of a %d-char pair\n",
                   cfg.words.size(), nfirst_words, cfg.target_len);
            printf("prefixes: %llu effective patterns from %d device patterns"
                   " (expected ~%.3g keys per match)\n",
                   (unsigned long long)effective_patterns, cfg.nprefixes, expected);
            printf("stage 1 : %.3g of candidates reach the word check"
                   " (shortest pattern %d chars)\n",
                   stage1_sum, shortest_len);
            if (stage1_sum > 1e-4)
                printf("  WARNING: stage 1 fires often - the shortest patterns dominate the\n"
                       "           cost. Drop the shortest words to get the throughput back.\n");
        } else if (cfg.prefixes.size() == 1) {
            printf("prefix  : %s (expected ~%.3g keys per match)\n", cfg.prefixes[0].c_str(),
                   expected);
        } else {
            printf("prefixes: %zu patterns, any match wins (expected ~%.3g keys per match)\n",
                   cfg.prefixes.size(), expected);
            if (cfg.prefixes.size() < 20) {
                for (const auto& s : cfg.prefixes)
                    printf("            %s\n", s.c_str());
            }
        }

        if (banner_plan.use_bloom) {
            printf("filter  : keys on ");
            for (int i = 0; i < banner_plan.ngroups; i++)
                printf("%s%d", i ? "+" : "", banner_plan.key_len[i]);
            printf(" chars (%d probe%s, %.0f entries), ~%.2g of candidates survive\n",
                   banner_plan.ngroups, banner_plan.ngroups == 1 ? "" : "s",
                   banner_plan.insertions, banner_plan.pass);
            if (banner_plan.pass > 1e-3)
                printf("  WARNING: that survivor rate costs throughput - each one pulls a whole\n"
                       "           warp into the pattern scan. The shortest patterns are the\n"
                       "           cause; dropping them is worth far more than their coverage.\n");
        }

        uint32_t nb = (iters + (uint32_t)batch - 1) / (uint32_t)batch;
        if (nb == 0)
            nb = 1;
        uint32_t launch_iters = nb * (uint32_t)batch;
        printf("threads : %llu total, %u cand/thread/launch, batch %d (%.1fM keys/launch combined)\n",
               (unsigned long long)T_total, launch_iters, batch,
               (double)T_total * launch_iters / 1e6);
    }

    std::atomic<int> global_found{0};
    std::atomic<int> g_crosscheck_ok{0};
    std::atomic<uint64_t> g_total_keys{0};
    std::atomic<bool> g_stop{false};
    std::mutex out_mtx;
    std::vector<BenchResult> bench_results(devices.size());
    std::vector<std::string> g_found_addrs;
    auto t_start = std::chrono::steady_clock::now();

    /* unique-ish id for this run's status log, so concurrent instances don't
       overwrite each other's uploaded snapshot */
    uint8_t sid_bytes[4];
    os_random(sid_bytes, sizeof(sid_bytes));
    std::string run_id = hex(sid_bytes, sizeof(sid_bytes));

    /* single reporter thread prints one combined progress line for every GPU
       together, instead of each device's worker printing its own; it also
       uploads a periodic status snapshot when --upload-status-per-30min is
       set, so progress survives the instance being stopped */
    const double status_interval = 1800.0; /* 30 min */
    std::thread reporter([&]() {
        auto t_last = t_start;
        double next_upload_el = status_interval;
        while (!g_stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            auto now = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(now - t_start).count();

            if (cfg.upload_status && !benchmark && el >= next_upload_el) {
                uint64_t total = g_total_keys.load();
                int found = global_found.load();
                std::vector<std::string> addrs_snapshot;
                {
                    std::lock_guard<std::mutex> lk(out_mtx);
                    addrs_snapshot = g_found_addrs;
                }
                std::string text = build_status_text(el, total, found, want, keep_forever,
                                                     devices.size(), cfg.prefixes, addrs_snapshot);
                upload_status_log(outdir, "gpuonion-status-" + run_id + ".txt", text);
                next_upload_el += status_interval;
            }

            if (std::chrono::duration<double>(now - t_last).count() < 2.0)
                continue;
            uint64_t total = g_total_keys.load();
            double rate = el > 0.0 ? (double)total / el : 0.0;
            std::lock_guard<std::mutex> lk(out_mtx);
            if (benchmark)
                printf("\r%s | total %llu | elapsed %.0fs / %.0fs  ",
                       format_rate(rate).c_str(), (unsigned long long)total, el, bench_secs);
            else
                printf("\r%s | total %llu | elapsed %.0fs | ~%.0f%% of expected  ",
                       format_rate(rate).c_str(), (unsigned long long)total, el,
                       100.0 * (double)total / expected);
            fflush(stdout);
            t_last = now;
        }
    });

    std::vector<std::thread> workers;
    workers.reserve(devices.size());
    for (size_t i = 0; i < devices.size(); i++)
        workers.emplace_back(run_device, devices[i], std::cref(cfg), std::ref(global_found),
                             std::ref(g_total_keys), std::ref(g_crosscheck_ok), t_start,
                             std::ref(out_mtx), std::ref(g_found_addrs), &bench_results[i]);
    for (auto& w : workers)
        w.join();

    g_stop.store(true);
    reporter.join();

    int ok = g_crosscheck_ok.load();
    if ((size_t)ok < devices.size())
        fprintf(stderr, "WARNING: GPU/CPU cross-check failed on %zu of %zu GPU(s); see errors above\n",
                devices.size() - (size_t)ok, devices.size());
    else
        printf("\nGPU/CPU cross-check OK (%d/%zu GPUs)\n", ok, devices.size());

    /* combined benchmark summary across all GPUs: throughput adds, so the
       combined rate is the sum of each device's own keys/sec (not total
       keys / total elapsed, since devices don't finish at the exact same
       instant) */
    if (benchmark) {
        double total_rate = 0.0;
        uint64_t total_keys = 0;
        for (const auto& r : bench_results) {
            if (r.elapsed > 0.0)
                total_rate += (double)r.keys / r.elapsed;
            total_keys += r.keys;
        }
        printf("\nbenchmark: %s combined across %zu GPU%s (%llu keys total)\n",
               format_rate(total_rate).c_str(), devices.size(), devices.size() == 1 ? "" : "s",
               (unsigned long long)total_keys);
        printf("expected time per match at this rate:\n");
        for (int len = 5; len <= 9; len++) {
            double secs = pow(32.0, (double)len) / total_rate;
            if (secs < 120.0)
                printf("  %d chars: %.1f s\n", len, secs);
            else if (secs < 7200.0)
                printf("  %d chars: %.1f min\n", len, secs / 60.0);
            else if (secs < 172800.0)
                printf("  %d chars: %.1f hours\n", len, secs / 3600.0);
            else
                printf("  %d chars: %.1f days\n", len, secs / 86400.0);
        }
    }

    return 0;
}
