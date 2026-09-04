# GPUonion: The FASTEST GPU-based Onion Address Vanity

A CUDA tool that searches for vanity addresses for Tor v3 onion services.
It implements on the GPU the same scalar-increment approach used by
[mkp224o](https://github.com/cathugger/mkp224o) and
[onion-vanity-address](https://github.com/offset/onion-vanity-address).

Up to 50 GKey/s @ 8x RTX 5090 ($3.5/hr on [vast.ai](https://cloud.vast.ai/?ref_id=314674)), which is 50x faster than 1.0 GKey/s @ 2x EPYC 128-Core Emb (512 threads) on mkp224o.

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
```

See [Optimize.md](Optimize.md) for the optimization history and the effect of each technique.

Example:

```bash
./gpuonion claude
```

Benchmark (measures with a 16-character prefix that never matches, and also prints the expected time per prefix length):

```bash
./gpuonion -b
```

Output is saved under `found/<address>/` in a format Tor can read directly:

- `hostname`
- `hs_ed25519_secret_key` (place it in `HiddenServiceDir` to use it)
- `hs_ed25519_public_key`

## Performance guide

| GPU | Speed |
|---|---|
| RTX 4070 (sm_89) | ~2.7 GKey/s (sustained) |
| H100 (sm_90) | ~5-6 GKey/s (estimated from the INT throughput ratio) |

The expected number of trials is `32^len`. Measured on an RTX 4070, a 6-character
prefix takes ~0.4 s on average, 7 characters ~13 s, 8 characters ~7 min, and
9 characters ~3.6 hours.

The implementation uses Montgomery x-only differential addition (3M+2S per
candidate) + batch inversion + 4×64-bit limbs with lazy reduction (PTX madc
chains). Details in [Optimize.md](Optimize.md).

## Notes

- Only base32 characters (`a-z`, `2-7`) are allowed in the prefix. `0 1 8 9` are not
- Handle the secret key files with care (never commit `found/`)
