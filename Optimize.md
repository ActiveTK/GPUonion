# GPUonion Optimization Log

Environment: RTX 4070 (sm_89, 46 SMs, 12GB), CUDA 12.8, Windows 11.
All measurements use `gpuonion -b` (1 warm-up run + 20-second measurement).

## Baseline (v1): 68.7 MKey/s

Cost per candidate in v1:

- Generic point addition (add-2008-hwcd-3): 9M
- **Inversion of Z (Fermat, 254sq + 11mul): ~265M ← 95% of the total**
- Affine conversion + serialization: 2M + tobytes ×2

The inversion dominates. Eliminating it is the first main target.

## Round 1: Montgomery batch inversion + mixed add + limb0 comparison

**68.7 → 952.8 MKey/s (13.9x)**

Changes (applied cumulatively, not measured individually):

1. **Montgomery batch inversion** — each thread accumulates (Y, Z, ΠZ) for
   B=32 candidates in local memory and amortizes one inversion over B candidates.
   Per candidate: ΠZ accumulation 1M + inversion ~165M/B + unwinding 2M + y=Y·zi 1M
2. **Dedicated fe_sq** — squaring needs 15 wide multiplies (mul needs 25). The
   254 squarings in the inversion chain get ~40% cheaper
3. **Mixed addition** — the step point (8T)·B is fixed, so it is normalized to
   affine and (Sy+Sx, Sy−Sx, 2d·Sx·Sy) is cached in constant memory. 9M → 7M
4. **Removal of the x coordinate and sign-bit computation** — the sign bit of
   the pubkey (bit 255) is never used for prefix matching (the longest prefix is
   30 characters = 150 bits). Matching only needs y, so recovering x
   (1M + tobytes) is dropped entirely. The full key is recomputed on the CPU
   when a match is found, so nothing is lost
5. **Direct limb0 comparison** — the low 51 bits of y (10.2 base32 characters)
   are exactly limb0 of the 5×51-bit representation. For prefixes ≤10
   characters, matching is a single 64-bit operation `(y0 ^ target) & mask`.
   Longer prefixes and the ~2^-50 false positives fall back to a full
   byte-wise comparison

Kernel info (ptxas): 172 registers, 0 spills, stack = 120B × B.

## Round 2: Parameter sweep

Starting from the defaults (B=32, tpb=256, blocks=368=SM×8):

| Change | MKey/s |
|---|---|
| B=8 | 570.8 |
| B=16 | 826.0 |
| B=32 (baseline) | 946.3 |
| B=64 | 1085.0 |
| B=128, tpb=128, blocks=736 | 1234.9 |
| B=256, tpb=128, blocks=736 | 1308.2 |
| **B=512, tpb=128, blocks=736** | **1331.5** |

Findings:

- **Bigger B wins** (inversion amortization outweighs the extra local memory),
  but with diminishing returns: +6% from 128→256, +1.8% from 256→512
- tpb=128 beats 256 (with 172 registers, 128×3 blocks = 384 resident threads/SM,
  which is higher occupancy than 256×1 = 256 threads)
- blocks is almost insensitive between 368 and 1472

### Failure: buying occupancy with -maxrregcount

- `-maxrregcount=128`: 1170.8 MKey/s (slower than the plain 1234.9)
- `-maxrregcount=96`: 1046.6 MKey/s

Squeezing registers to raise occupancy costs more in spill traffic to local
memory than it gains. **Rejected**.

### Rejected: warp-cooperative grand inversion

Considered "combine the products of 32 lanes with shuffles into a single
inversion", but under SIMT, 32 lanes redundantly computing the same inversion
costs the same issue slots as one lane. In other words the per-thread batch
inversion (each lane inverting its own value in parallel) is already perfect
SIMD amortization, and warp cooperation has **no theoretical gain at all**.
Noticed before implementing; never implemented.

## Round 3: Montgomery x-only differential addition — 1331.5 → 1619.1 MKey/s (+21.6%)

A CUDA version of the "compute only the y coordinate" approach claimed by the
reference repository.

- The Edwards y and the Curve25519 u are related by the birational map
  u = (1+y)/(1−y) (RFC 7748). In projective coordinates
  **u = (Z+Y)/(Z−Y), so the conversion needs no inversion**
- Each step is the xADD (differential addition) of the Montgomery ladder:
  P_{k+1} = P_k + S computed from P_{k−1} = P_k − S. **4M + 2S**
  (down from the 7M of Edwards mixed add; with fe_sq counted as 0.7M,
  7M → 5.4M equivalent)
- The candidate y is (U−W)/(U+W). The denominator D = U+W goes through the batch inversion
- Thread state is (U, W) plus the previous (Um, Wm): 4 field elements
- At initialization, P_t − S is computed in addition to P_t (just pass −S to ge_add)
- The x coordinate disappears entirely, so the sign bit is lost, but it is not
  needed for matching (Round 1 #4), and the host recomputes everything from the
  scalar on a hit, so this is fine

A/B under identical conditions (B=512, tpb=128, blocks=736):

| Kernel | MKey/s |
|---|---|
| Edwards extended (--ext) | 1341.1 |
| Montgomery x-only | **1619.1** |

## Round 4: Removing carries from the hot path — 1619.1 → 1656.7 MKey/s (+2.3%)

In the 5×51 representation, limbs up to < 2^54 are safe as inputs to
fe_mul/fe_sq (the carry extraction (hi<<13) of the 128-bit accumulator does not
break). The 4 additions/subtractions inside xADD were replaced with
fe_add_nc / fe_sub_nc, which do no carry propagation.

### Failure: #pragma unroll 2 on the pass1 loop

Hoped that the state shift in xADD (fe_copy ×4) could be eliminated by register
renaming, but 1656.7 → 1625.7 MKey/s (−1.9%). The larger code apparently hurt
the I-cache/scheduler. **Rejected** (reverted to unroll 1).

### Follow-up sweep (as of Round 4, mont 5x51)

- tpb: 64:1633 / 96:1584 / **128:1640** / 192:1587 / 256:1538
- blocks: almost flat from 368 to 2944 (±2%)
- iters: almost flat from 512 to 4096
- → tpb=128, blocks=SM×16, iters=1024 became the defaults. Parameters have plateaued

### Abandoned: bottleneck identification with Nsight Compute

`ERR_NVGPUCTRPERM` — GPU performance counters require an administrator-level
configuration change, so this was abandoned. From here on, decisions are based
on empirical A/B measurements.
(Side observation: while ncu was running, clock thermal drift was observed,
1640→1580. The 20-second benchmark numbers fluctuate by about ±1.5%)

## Round 5: PTX madc chains — 1656.7 → 1757.1 MKey/s (+6.1%)

fe_mac (multiply-accumulate into a 128-bit accumulator) was changed from C++
comparison-based carries (`hi += (lo < prev)`) to inline PTX
`mad.lo.cc.u64 / madc.hi.u64`, and carry additions to `add.cc / addc`.
The SETP/SEL instruction sequences that were left to ptxas become hardware
carry-flag chains.

## Round 6: 4x64 limb representation (fe4) — 1757.1 → 2158.0 MKey/s (+22.8%)

Switched to the standard layout used by VanitySearch-style GPU bignums:

- **fe4 = 4×64-bit limbs, lazy reduction mod 2^256−38 (=2p)**, values in [0, 2^256)
- Multiplication: 4×4 schoolbook (PTX madc chains, 512-bit intermediate) →
  fold hi×38. 16+4 wide multiplies (down from 25 in 5×51), all carries in CC-flag chains
- Addition/subtraction: 4-limb chain + 38 correction (implemented branch-free
  after proving that the overflow/borrow wrap-around always converges within 2 rounds)
- Even with arithmetic mod 2p, y = N/D mod p is obtained correctly by a final
  canonicalization (conditional subtraction of 2p, then p). The Fermat inversion
  a^(p−2) can also run mod 2p as-is (the residue mod p is the same)
- Only the y used for matching is canonicalized. The limb0 comparison is even
  more natural with 4×64 (the low 51 bits of y[0] are used directly)
- The local-memory batch arrays also shrink from 40B → 32B per field element
  (traffic −20%)
- A portable C++ fallback is implemented on the host side, and the selftest
  verifies that mul/add/sub/invert agree with the 5×51 implementation

**Registers dropped from 168 → 128**, raising resident threads per SM from
384 → 512. Compact carry chains also help with register pressure.

Bugs found and fixed during implementation (in my own first draft):
- Missing carry in the last row of the schoolbook (p7 propagation lost in the lo chain)
- Final carry dropped in the fold (loses 2^256 with probability ~2^-245).
  Made exact with one extra fold, using the property that the value after
  wrapping is always tiny

## Round 7: Aggressive reduction of the fe4 kernel — 2158.0 → 2860.4 MKey/s (+32.5%)

Three changes landed at once:

1. **Dedicated fe4_sq** — accumulate the 6 cross products, double the whole
   512-bit value, then add the 4 squares in a single madc chain. Wide
   multiplies 16 → 10
2. **Projective scaling of xADD (the κ trick)** — the two constant
   multiplications in xADD, t1=N·(u_S+1) and t2=D·(u_S−1), can both be scaled
   by (u_S+1)^{-1} and (U:W) stays projectively identical. With
   κ = (u_S−1)/(u_S+1) precomputed, t1 = N (the multiplication vanishes!) and
   t2 = D·κ, so **xADD goes from 4M+2S → 3M+2S**
3. **Batch arrays 3 → 2** — if pass1 builds M_k = N_k·C_{k−1} (same number of
   multiplications), the backward pass is just y = M_k·inv and inv ×= D_k, and
   the C array is no longer needed at all. Local memory round trip
   192B → 128B per candidate

Along the way, an asm operand-numbering mistake in fe4_sq (still %6.. after
copy-pasting from fe4_mul) made nvcc emit an ICE. Fixed by correcting the
operand numbers.

Theoretical cost per candidate: **7 mul + 2 sq = about 160 wide multiplies**
(v1 was effectively ~280 fe_mul ≈ 7000 wide multiplies including the
inversion; about 1/44)

### Reference: batch-size re-sweep (fe4, tpb=128)

B=256: 2675 / B=512: 2690 / B=1024: 2773 MKey/s — almost flat.
Consecutive benchmarks fluctuate by ±3–5% (clock boost state), so parameter
differences beyond this are buried in noise. Default set to B=512.

### Reconsidered: warp-cooperative grand inversion

Dismissed in Round 2 as "theoretically pointless", but that turned out to be
half wrong. It has zero effect on making the inversion faster (each lane
inverting its own value in parallel under SIMD is already optimal), but it is
useful for "shrinking B so local memory fits in L2 while keeping the
amortization ratio" (the inversion of the grand product is a single
warp-shared value, so the redundant computation is free). However, since the
flat measurements at B=256–1024 indicate we are not DRAM-bound, it was
**deferred**.

### Rejected: merging via the 2-torsion point

Using the fact that u → 1/u is "translation by (0,0)" on the Montgomery curve,
matching P+T2 (the point whose y becomes −y) for a candidate P comes almost for
free, which looks like doubling the throughput. But T2 lies outside the
prime-order subgroup, and P+T2 = aB + T2 has no discrete logarithm with base B
(i.e. no signing-capable secret key exists). **Fundamentally unusable**. An
interesting trap.

### Rejected: Karatsuba / 8×32-bit limbs

- Karatsuba (4×64): the extra additions/subtractions and sign handling eat the
  efficiency of the madc schoolbook. Judged unfavorable at this word size, as
  is conventional
- 8×32-bit: ptxas already decomposes 64-bit madc into IMAD.WIDE (32×32), which
  yields effectively the same structure. No point rewriting by hand

## Round 8: Lazy 3-way comparison

The canonicalization of y (2 passes of conditional 2p/p subtraction) is
deferred until a match hits. The lazy y is y_true + jp (j∈{0,1,2}), so limb0 is
determined as y0+19j (mod 2^64); prefix candidates are picked up with 3 64-bit
comparisons and only then canonicalized. The effect is within measurement error
(2708 vs 2690 MKey/s), but it cannot lose theoretically, so it was adopted.

## Round 9: Failed occupancy experiments

### Failure: maxrregcount=96 (fe4)

Tried to squeeze 128 registers → 96 to raise occupancy from 512→640
threads/SM, but **2690 → 814 MKey/s (−70%)**. Spills hit local memory traffic
directly. Far worse than in the 5×51 era (−5–15%); for the fe4 kernel, 128
registers is exactly the equilibrium point.

### Failure: 2-way interleaving (--x2)

The idea: one physical thread advances two independent chains to double ILP.
The SASS showed IMAD:IADD3 ≈ 1:1 with an estimated INT pipe utilization of
~50%, so it looked promising, but **2316 MKey/s (−14%)** (2133 at B=256).
Doubling the state and arrays raised register/cache pressure, and the loss in
occupancy outweighed the ILP gained. The implementation was removed (only this
record remains).

## Final results

| Stage | MKey/s | Cumulative speedup |
|---|---:|---:|
| v1 naive implementation (Fermat inversion per candidate) | 68.7 | 1.0x |
| + Montgomery batch inversion / fe_sq / mixed add / limb0 comparison | 952.8 | 13.9x |
| + parameter tuning (B=512, tpb=128) | 1331.5 | 19.4x |
| + Montgomery x-only differential addition | 1619.1 | 23.6x |
| + carry removal in the hot path | 1656.7 | 24.1x |
| + PTX madc chains | 1757.1 | 25.6x |
| + 4×64 limbs (fe4) + lazy reduction | 2158.0 | 31.4x |
| + dedicated fe4_sq + κ trick + 2 arrays | **~2716 (sustained)** | **39.5x** |

- Final sustained speed: **about 2.72 GKey/s (RTX 4070)**. Instantaneous values up to 2.86 GKey/s observed
- A 7-character prefix (expected 34.4G keys) was found in a measured 23.3 seconds
- Expected time for an 8-character prefix: about 6.7 minutes / 9 characters: about 3.6 hours
- Generated keys were independently verified at every stage with a pure
  big-integer ed25519 implementation in Python
  (scalar·B = pubkey, address derivation, clamp shape)

## Expected scaling to H100

- The kernel is INT32 (IMAD/IADD3) bound. H100 SXM has 132 SMs × 64 INT32/clk
  × ~1.8GHz ≈ about 2.0x the INT throughput of an RTX 4070 (46 SMs × 2.7GHz)
  → **a simple estimate gives 5–6 GKey/s per GPU**
- The inline PTX asm (mad/madc/add.cc) is architecture-independent. The fatbin
  already bundles an sm_90 native binary + compute_90 PTX, so Blackwell and
  later run via JIT
- Local memory usage: 32KB/thread at B=512. With the large L2 (50MB) and HBM
  bandwidth of the H100, B=1024 or more may be even better (needs measurement: tune with `-B`)
- Multi-GPU is designed to run as parallel processes with devices selected via `-d`

## Summary of things tried and not adopted / decided against

| Idea | Verdict |
|---|---|
| Buying occupancy with -maxrregcount | Measured much worse. Rejected |
| #pragma unroll 2 | Measured −1.9%. Rejected |
| Warp-cooperative grand inversion | Theoretically ineffective as an inversion speedup. Useful for shrinking B, but deferred since not DRAM-bound |
| y-only (Edwards) differential addition | Derived, but 6S+8M is heavier than mixed add. Montgomery x-only wins |
| Doubling candidates via 2-torsion merging | No secret key exists outside the subgroup; fundamentally impossible |
| Karatsuba / 8×32 limbs | Expected to be no better than the madc schoolbook. Not implemented |
| Using the FP32 pipe as well (float limbs) | FP64 on Ada runs at 1/64 rate, making correctness hard to guarantee. Deferred |
| 2-way interleaving | Measured −14%. Removed |
| Nsight Compute profiling | Abandoned for lack of permissions (requires administrator configuration) |
