# CuOzaki accelerator

Emulating double precision matrix multiplication on GPU **int8 tensor cores** with the
**Ozaki scheme II** (integer / modular variant). A learning project: C++20, CUDA, CCCL
(Thrust, libcu++).

A double-precision GEMM is turned into `N` small-modulus int8 GEMMs, whose results are
combined exactly with the Chinese Remainder Theorem (CRT). The number of moduli `N` is the
accuracy / speed knob.

## How it works

Given `C = A * B` (A is `M x K`, B is `K x N`):

1. **Scale.** Every row of A and every column of B gets a power-of-two scale `2^mu_i` /
   `2^nu_j`, taken from the exponent of its largest element, so that the scaled values are
   integers below `2^b`. `b` is chosen so the integer product cannot wrap around the CRT
   modulus: `2 * K * 4^b < P`, with `P` the product of all moduli.
2. **Integers.** `A' = trunc(A * 2^-mu)` and `B' = trunc(B * 2^-nu)` (int64).
3. **Residues.** For each modulus `p_l <= 256`, reduce `A'` and `B'` mod `p_l`, centred into
   `[-p/2, p/2)` so that they fit in int8.
4. **int8 GEMMs.** For each modulus, multiply the residue matrices on the tensor cores
   (`mma.sync.m16n8k32`, int8 x int8 -> int32). The sums are exact.
5. **CRT.** Combine the `N` products into the integer product `A'B'` (128-bit arithmetic),
   centre it, and undo the scaling: `C = D_mu * (A'B') * D_nu`. The scale is a power of two,
   so this step is exact.

The only roundings are the truncation in step 2 and the final conversion to double, which is
why enough moduli give a result that is at least as accurate as native DGEMM.

### Code layout

```
include/ozaki/ozaki_cpu.hpp   CPU pipeline, one function per step (reference implementation)
src/ozaki_cpu.cpp             pure C++ library `ozaki_cpu`
include/ozaki/kernels.cuh     device code: exponent, integer conversion, residue, int8 GEMM, CRT kernels
include/ozaki/ozaki.cuh       host side: `ozaki::ogemm` class and kernel launchers (header-only)
bench/bench_gemm.cu           benchmark against cuBLAS DGEMM, with an accuracy measurement
```

`ogemm<T, M, N, K, numModuli>` owns all device buffers, so repeated calls reuse them:

```cpp
#include "ozaki/ozaki.cuh"          // include from .cu files only

ozaki::ogemm<double, 4096, 4096, 4096, 16> g;
g.run(hA, hB, hC);                  // hA: M x K row-major, hB: K x N column-major, hC: M x N row-major
// or, to time only the device work: g.upload(hA, hB); g.compute(); g.download(hC);
```

The int8 GEMM kernel uses a 64 x 64 output tile per block (4 warps, each owning a 32 x 32
part), stages 16-byte chunks of both operands through padded shared memory (no bank conflicts
on the fragment loads), and writes the result through shared memory with 16-byte stores.

Constraints: `M`, `N`, `K` are multiples of 64, `numModuli` is between 2 and 16, and B is
stored column-major.

## Build and run

Requirements: CUDA toolkit 13.x, CMake >= 3.31, a C++20 compiler, a GPU with int8 tensor
cores (the kernels target compute capability 8.9).

```
cmake --preset release
cmake --build --preset release -j
./build/release/bench/bench_gemm
```

## Results

NVIDIA GeForce RTX 4070 SUPER (compute capability 8.9, FP64 throttled to a small fraction of
FP32). `C = A * B` with `M = N = K = 4096`, entries uniform in `[-1, 1]`.

| | time (ms) | TFLOP/s | speedup | max error | mean error | max error (ulp) |
|---|---|---|---|---|---|---|
| cuBLAS DGEMM | 262.0 | 0.525 | 1.00x | 1.67 u | 0.26 u | 2027 |
| Ozaki II, 14 moduli (b = 48) | 49.0 | 2.81 | 5.35x | 3.42 u | 0.92 u | 8269 |
| Ozaki II, 15 moduli (b = 52) | 52.5 | 2.62 | 4.99x | 0.07 u | 0.02 u | 97.6 |
| Ozaki II, 16 moduli (b = 56) | 55.8 | 2.46 | 4.70x | 0.03 u | 0.01 u | 0.5 |

How to read it:

- Time is device work only (no host copies), averaged over 5 runs after 2 warm-up runs,
  measured with CUDA events. TFLOP/s counts `2 * M * N * K` flops for both columns.
- Accuracy is measured on 512 random output entries against a double-double reference
  (about 100 bits). "u" is the unit roundoff 2^-53; the error of an entry is
  `|c - ref| / sum_k |a_ik * b_kj|` in units of `u`, which does not blow up where the sum
  cancels. The ulp column is relative to the reference value itself, so it is large where the
  true result happens to be small, even for cuBLAS.
- Accuracy needs `b >= 53` to match double: with 14 moduli the emulation is less accurate than
  DGEMM, with 15 it is more accurate, and with 16 the worst sampled entry is 0.5 ulp off.
- Each additional modulus costs about 3.4 ms here (derived from the timings above), which is
  well below what the int8 tensor cores can do, so the GEMM kernel has room for optimisation.
- These numbers are for well-scaled data. Rows whose entries span many orders of magnitude
  lose accuracy under per-row scaling, and that case is not benchmarked yet.

## Status and next steps

- Done: CPU reference, GPU pipeline for double, benchmark with an accuracy metric.
- Next: FP32 and double-double emulation, an int4 variant, and optimisation of the int8 GEMM
  (pipelined loads, larger warp tiles, fused residue and CRT stages).

## References

- Ozaki, Ogita, Oishi, Rump: error-free transformation of matrix multiplication (2012).
- Ootomo, Ozaki, Yokota: DGEMM on integer matrix multiplication unit (2024).
- Ozaki scheme II (CRT-based variant) papers from 2025.
- NVIDIA PTX ISA: `mma.sync` shapes and types for int8.
