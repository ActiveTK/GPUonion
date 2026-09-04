# GPUonion: The FASTEST GPU-based Onion Address Vanity (thanks to my best friend Fable 5)

A CUDA tool that searches for vanity addresses for Tor v3 onion services.
It implements on the GPU the same scalar-increment approach used by
[mkp224o](https://github.com/cathugger/mkp224o) and
[onion-vanity-address](https://github.com/offset/onion-vanity-address).

Up to 50 GKey/s @ 8x RTX 5090 ($3.5/hr on [vast.ai](https://cloud.vast.ai/?ref_id=314674)), which is 50x faster than 1.0 GKey/s @ 2x EPYC 128-Core Emb (512 threads) and 500,000x faster than 100 Kkeys/s @ i5 3320M (My ThinkPad, 4 threads) on mkp224o.

| Length | Expected trials | Average time | Cost ($3.5/h) |
|---|---|---|---|
| 1 | 32 | instant | ≈ $0 |
| 2 | 1,024 | instant | ≈ $0 |
| 3 | 32,768 | instant | ≈ $0 |
| 4 | ~1.05×10⁶ | instant | ≈ $0 |
| 5 | ~3.36×10⁷ | instant | ≈ $0 |
| 6 | ~1.07×10⁹ | 21 ms | ≈ $0 |
| 7 | ~3.44×10¹⁰ | 0.69 s | $0.0007 |
| 8 | ~1.10×10¹² | 22 s | $0.02 |
| 9 | ~3.52×10¹³ | 11.7 min | $0.68 |
| 10 | ~1.13×10¹⁵ | 6.3 hours | $22 |
| 11 | ~3.60×10¹⁶ | 8.3 days | $700 |
| 12 | ~1.15×10¹⁸ | 267 days (0.73 yr) | $22,400 |
| 13 | ~3.69×10¹⁹ | 23.4 years | $717,000 |
| 14 | ~1.18×10²¹ | 748 years | $23 million |

## How to use

Ask fable `Yo my best friend fable plz tell me how to use ActiveTK/GPUonion?`

## Algorithm

- Starting from a random clamped scalar `a0`, the `i`-th candidate of thread `t` is
  `a = a0 + 8*(t + i*T)` (`T` = total number of threads)
- On the GPU, each step to the next candidate is a single Curve25519 (Montgomery)
  x-only differential addition (3M+2S); the full scalar multiplication is done
  only once, at initialization. The Edwards y is recovered through the
  birational map y = (U−W)/(U+W), and the denominators are amortized with a
  batch inversion
- Keeping the increment a multiple of 8 preserves the clamp structure, so the
  result is directly valid as a Tor expanded secret key (`hs_ed25519_secret_key`)
- Every found key is recomputed and verified on the CPU before it is written
  (the GPU and CPU share the same `__host__ __device__` code)

## Build

### Linux (production environments such as H100)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

By default the binary bundles native code for sm_60 through sm_90 plus
compute_90 PTX, so it runs on almost every GPU from Pascal onward
(Blackwell and later via JIT). If you only need a specific GPU, the build is
much faster:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90   # H100 only
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120  # for RTX 5090
```

### Windows (development environment)

VS 2022 + CUDA Toolkit 12.x:

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Usage

```bash
./gpuonion <prefix> [options]
./gpuonion -b [options]
  <prefix>       base32 prefix to search for (a-z 2-7)
  -b, --bench    run a ~20 second benchmark (no prefix needed)
  --split-pubkey <hex|file>          split-key search for someone else's key (see below)
  --split-keygen <dir>               requester: create secret/public half
  --split-combine <secret-half> <offset>   requester: build the final Tor key
  -d <spec>      CUDA device selection: "all" (default, use every GPU) / a single index / a comma-separated list such as "0,1,2"
                 each selected GPU runs concurrently on its own host thread
  -n <count>     stop after finding this many keys (default 1)
  -o <dir>       output directory (default ./found)
  --blocks <n>   number of blocks (default: SM count * 16)
  -t <threads>   threads per block (default 128)
  -i <iters>     candidates tried per thread per launch (default 1024)
  -B <batch>     batch size for the Montgomery batch inversion, power of two 8..1024 (default 128)
  --m51          use the 5x51 limb kernel (for A/B comparison)
  --ext          use the Edwards extended-coordinate kernel (for A/B comparison, slowest)
  --selftest     run the internal tests only
  and so on (see --help)
```

See [Optimize.md](Optimize.md) for the optimization history and the effect of each technique.

Example:

```bash
./gpuonion activetkkami
```

Benchmark (measures with a 16-character prefix that never matches, and also prints the expected time per prefix length):

```bash
./gpuonion -b
```

Output is saved under `found/<address>/` in a format Tor can read directly:

- `hostname`
- `hs_ed25519_secret_key` (place it in `HiddenServiceDir` to use it)
- `hs_ed25519_public_key`

## Split-key (have someone else's GPU search without giving them your key)

Renting a fleet of GPUs means the machine that finds the key is not yours. In
split-key mode the GPU never sees the secret key at all. The requester keeps a
random scalar `a_c` (the **secret half**) and hands out only `A_c = a_c*B` (the
**public half**). The worker searches for an offset `k` such that `A_c + k*B`
gives the wanted address and returns `k`. Since `a_c*B + k*B = (a_c + k)*B`,
only the requester can form the final secret scalar `a = (a_c + k) mod L`.
The offset is not sensitive: with the public half it yields only the public key,
which the address already reveals.

Step 1, requester (any machine, no GPU needed):

```bash
./gpuonion --split-keygen mykey
#  mykey/split_secret_half   <- keep private (Tor hs_ed25519_secret_key format)
#  mykey/split_public_half   <- 64 hex chars; this is all the worker gets
```

Step 2, worker (GPU; every other option works as usual):

```bash
./gpuonion activetk --split-pubkey <public half hex, or path to split_public_half>
#  found/<address>/split_offset           <- the offset k (hex), sent back to the requester
#  found/<address>/split_public_half      <- which public half it belongs to
#  found/<address>/hostname, hs_ed25519_public_key
```

There is no `hs_ed25519_secret_key` on the worker, and none can be derived
there. The offset file is what gets uploaded in place of the key (or kept
local with `--no-upload`).

Step 3, requester:

```bash
./gpuonion --split-combine mykey/split_secret_half <offset hex, or path to split_offset>
#  found/<address>/hostname
#  found/<address>/hs_ed25519_public_key
#  found/<address>/hs_ed25519_secret_key   <- ready for HiddenServiceDir
```

The combine step recomputes the address from `(a_c + k) mod L`, checks it
against `A_c + k*B`, and, when the offset is given as a file, against the
`hostname` next to it, so a wrong secret half is caught rather than written.
Tor reduces expanded secret scalars mod L before use (its own key blinding
produces such scalars), so the unclamped sum is a valid key as is. The search
speed is identical to normal mode: only the per-thread start point changes.

## Notes

- Only base32 characters (`a-z`, `2-7`) are allowed in the prefix. `0 1 8 9` are not
- Handle the secret key files with care (never commit `found/`)
