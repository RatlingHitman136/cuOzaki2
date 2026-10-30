#pragma once

#include <atomic>
#include <cstdint>
#include <cuda/std/cmath>
#include <concepts>
#include <cstddef>

namespace ozaki {

__extension__ typedef unsigned __int128 u128;

template <int N>
struct alignas(16) CrtConsts {
  u128 P;      // product of all moduli
  u128 Pl[N];  // P / p_l
  int p[N];    // moduli
  int q[N];    // (P / p_l)^-1 mod p_l
};

}  // namespace ozaki

namespace ozaki::kernels {

__device__ inline std::uint32_t ld32(const std::int8_t* p) {
  return *reinterpret_cast<const std::uint32_t*>(p);      // p must be 4-byte aligned
}

// c += a * b, one warp, 16x8x32 int8 -> int32. c is both input and output.
__device__ inline void mma_s8(int (&c)[4], const std::uint32_t (&a)[4], const std::uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <std::floating_point T, int thread_count>
__global__ void row_exp_k(const T* A, int* mu, int m, int k, int opbits) {
  const size_t blk_i = blockIdx.x;
  const size_t thd_i = threadIdx.x;

  T max_val = 0;
  for (size_t in_row_i = thd_i; in_row_i < k; in_row_i += blockDim.x) {
    T cur = cuda::std::fabs(A[blk_i * k + in_row_i]);
    if (cur > max_val) {
      max_val = cur;
    }
  }

  constexpr int num_warps = thread_count / 32;          // 128 -> 4
  __shared__ T warp_max[num_warps];

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;

  for (int off = 16; off > 0; off >>= 1)
    max_val = cuda::std::max(max_val, __shfl_down_sync(0xffffffff, max_val, off));
  if (lane == 0) warp_max[warp] = max_val;
  __syncthreads();

  if (warp == 0) {
    T v = lane < num_warps ? warp_max[lane] : T{0};     // 0 is the identity for max of |a|
    for (int off = 16; off > 0; off >>= 1)
      v = cuda::std::max(v, __shfl_down_sync(0xffffffff, v, off));
    if (lane == 0) {
      int e;
      cuda::std::frexp(v, &e);
      mu[blockIdx.x] = e - opbits;
    }
  }
}

template <std::floating_point T>
__global__ void to_integer_rows_k(const T* A, std::int64_t* Ap, int* mu, int k) {
  const size_t blk_i = blockIdx.x;
  const size_t thd_i = threadIdx.x;
  int cur_mu = mu[blk_i];

  for (size_t in_row_i = thd_i; in_row_i < k; in_row_i += blockDim.x) {
    T cur = A[blk_i * k + in_row_i];
    Ap[blk_i * k + in_row_i] = static_cast<std::int64_t>(cuda::std::trunc(cuda::std::ldexp(cur, -cur_mu)));
  }
}

template <int numModuli>
__global__ void residues_k(const std::int64_t* Xp, std::int8_t* Xres, std::size_t count, CrtConsts<numModuli> crt) {
  const size_t idx = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const size_t stride = size_t{gridDim.x} * blockDim.x;

  for (int curMod = 0; curMod < numModuli; ++curMod) {
    int p = crt.p[curMod];

    for (size_t i = idx; i < count; i += stride) {
      auto cur = Xp[i] % p;
      cur += cur < 0 ? p : 0;
      cur -= cur >= (p + 1)/2 ? p : 0;

      Xres[curMod * count + i] = static_cast<std::int8_t>(cur);
    }
  }
}

template <std::floating_point T, int numModuli, int BM = 64, int BN = 64, int BK = 64, int TM = 32, int TN = 32, int TK = 32>
__global__ void int8_gemm_k(const std::int8_t* Apres, const std::int8_t* Bpres, std::int32_t* Cprod, int m, int n, int k) {
  static_assert(BM % TM == 0 && BN % TN == 0 && TM % 16 == 0 && TN % 8 == 0, "tile sizes");
  static_assert(TK == 32 && BK % TK == 0 && BK % 16 == 0, "m16n8k32 needs k steps of 32");

  const size_t bx = blockIdx.x;
  const size_t by = blockIdx.y;

  const size_t tcount = blockDim.x;
  const size_t ti = threadIdx.x;

  constexpr int kChunk = 16;
  constexpr int kChunksPerRow = BK / kChunk;
  constexpr int kChunksM = BM * kChunksPerRow;
  constexpr int kChunksN = BN * kChunksPerRow;

  // 16 bytes of padding per row
  constexpr int LD = BK + 16;

  constexpr int kWarpsN = BN / TN;
  constexpr int kMmaM = TM / 16;
  constexpr int kMmaN = TN / 8;

  // shared block mem
  __shared__ alignas(16) std::int8_t bA[BM * LD];
  __shared__ alignas(16) std::int8_t bB[BN * LD];
  __shared__ alignas(16) std::int32_t bC[BM * BN];

  // this thread's place in the block tile
  const int warp = ti / 32;
  const int lane = ti % 32;
  const int warp_m = warp / kWarpsN;
  const int warp_n = warp % kWarpsN;

  const int g = lane >> 2;   // row group of the mma fragments, 0..7
  const int t = lane & 3;    // position within the group, 0..3

  for (int curMod = 0; curMod < numModuli; ++curMod) {
    const std::int8_t* Ar = Apres + size_t(curMod) * m * k;
    const std::int8_t* Br = Bpres + size_t(curMod) * n * k;

    int acc[kMmaM][kMmaN][4] = {};

    for (int i = 0; i < k/BK; i++) {
      // populate bA
      for (int in_bA_i = ti; in_bA_i < kChunksM; in_bA_i += tcount) {
        const int in_bA_x = (in_bA_i % kChunksPerRow) * kChunk;
        const int in_bA_y = in_bA_i / kChunksPerRow;
        const uint4 v = *reinterpret_cast<const uint4*>(&Ar[(by * BM + in_bA_y) * k + i * BK + in_bA_x]);
        *reinterpret_cast<uint4*>(&bA[in_bA_y * LD + in_bA_x]) = v;
      }

      // populate bB
      for (int in_bB_i = ti; in_bB_i < kChunksN; in_bB_i += tcount) {
        const int in_bB_x = (in_bB_i % kChunksPerRow) * kChunk;
        const int in_bB_y = in_bB_i / kChunksPerRow;
        const uint4 v = *reinterpret_cast<const uint4*>(&Br[(bx * BN + in_bB_y) * k + i * BK + in_bB_x]);
        *reinterpret_cast<uint4*>(&bB[in_bB_y * LD + in_bB_x]) = v;
      }
      __syncthreads();

      // block gemm
      #pragma unroll
      for (int kk = 0; kk < BK; kk += TK) {
        std::uint32_t a[kMmaM][4], b[kMmaN][2];

        #pragma unroll
        for (int mi = 0; mi < kMmaM; ++mi) {
          const int r = warp_m * TM + mi * 16 + g;
          a[mi][0] = ld32(&bA[(r    ) * LD + kk + 4 * t     ]);
          a[mi][1] = ld32(&bA[(r + 8) * LD + kk + 4 * t     ]);
          a[mi][2] = ld32(&bA[(r    ) * LD + kk + 4 * t + 16]);
          a[mi][3] = ld32(&bA[(r + 8) * LD + kk + 4 * t + 16]);
        }

        #pragma unroll
        for (int ni = 0; ni < kMmaN; ++ni) {
          const int c = warp_n * TN + ni * 8 + g;
          b[ni][0] = ld32(&bB[c * LD + kk + 4 * t     ]);
          b[ni][1] = ld32(&bB[c * LD + kk + 4 * t + 16]);
        }

        #pragma unroll
        for (int mi = 0; mi < kMmaM; ++mi)
          #pragma unroll
          for (int ni = 0; ni < kMmaN; ++ni)
            mma_s8(acc[mi][ni], a[mi], b[ni]);
      }
      __syncthreads();
    }

    // stage the accumulators in bC (row-major BM x BN), then write the tile with 16-byte stores
    #pragma unroll
    for (int mi = 0; mi < kMmaM; ++mi)
      #pragma unroll
      for (int ni = 0; ni < kMmaN; ++ni) {
        const int row = warp_m * TM + mi * 16 + g;
        const int col = warp_n * TN + ni * 8 + 2 * t;
        bC[(row    ) * BN + col    ] = acc[mi][ni][0];
        bC[(row    ) * BN + col + 1] = acc[mi][ni][1];
        bC[(row + 8) * BN + col    ] = acc[mi][ni][2];
        bC[(row + 8) * BN + col + 1] = acc[mi][ni][3];
      }
    __syncthreads();

    std::int32_t* Cp = Cprod + size_t(curMod) * m * n;
    constexpr int kChunksPerCRow = BN / 4; 
    for (int in_bC_i = ti; in_bC_i < BM * kChunksPerCRow; in_bC_i += tcount) {
      const int in_bC_x = (in_bC_i % kChunksPerCRow) * 4;
      const int in_bC_y = in_bC_i / kChunksPerCRow;
      const uint4 v = *reinterpret_cast<const uint4*>(&bC[in_bC_y * BN + in_bC_x]);
      *reinterpret_cast<uint4*>(&Cp[(by * BM + in_bC_y) * n + bx * BN + in_bC_x]) = v;
    }
  }
}


template <std::floating_point T, int numModuli>
__global__ void crt_k(const std::int32_t* Cprod, const int* mu, const int* nu, T* C, int m, int n, CrtConsts<numModuli> crt) {
  const size_t count = size_t(m) * n;
  const size_t idx = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const size_t stride = size_t{gridDim.x} * blockDim.x;

  for (size_t i = idx; i < count; i += stride) {
    u128 s = 0;
    for (int l = 0; l < numModuli; ++l) {
      const int p = crt.p[l];

      int c = Cprod[l * count + i] % p; 
      c += c < 0 ? p : 0;               
      const int tl = c * crt.q[l] % p;

      s += static_cast<u128>(tl) * crt.Pl[l]; 
      s -= s >= crt.P ? crt.P : 0;            
    }

    const double val = s > crt.P / 2 ? -static_cast<double>(crt.P - s) : static_cast<double>(s);

    const size_t row = i / n;
    const size_t col = i % n;
    C[i] = static_cast<T>(cuda::std::ldexp(val, mu[row] + nu[col]));
  }
}

}  // namespace ozaki::kernels
