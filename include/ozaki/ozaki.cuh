#pragma once

// Host-side entry point: error checking, helpers, launchers and the ogemm class.
// Header-only and compiled by nvcc, so include it from .cu files only. The kernels
// themselves are in kernels.cuh.

#include <cassert>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cuda/cmath>
#include <cuda/stream_ref>
#include <span>
#include <vector>
#include <array>
#include <algorithm>

#include "ozaki/kernels.cuh"

namespace ozaki {

inline constexpr std::array<int, 16> kModuli{256, 255, 253, 251, 247, 241, 239, 233,
                                             229, 227, 223, 211, 199, 197, 193, 191};


inline void check(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
    std::exit(EXIT_FAILURE);
  }
}

template <std::size_t N>
consteval int operand_bits(const std::array<int, N>& moduli, int k) {
  unsigned __int128 P = 1;
  for (int p : moduli) P *= p;
  const unsigned __int128 q = (P - 1) / (2u * k);
  int b = 0;
  while ((q >> (2 * (b + 1))) != 0) ++b;
  return b;
}

#define CUDA_CHECK(call) ::ozaki::check((call), __FILE__, __LINE__)

// Compute capability of the current device as major * 10 + minor (89 for Ada).
inline int compute_capability() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int major = 0, minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
  return major * 10 + minor;
}

template <typename T>
void launch_row_exponents_impl(const T* dA, int* mu, int m, int k, int opbits, cudaStream_t s) {
  kernels::row_exp_k<T, 128><<<m, 128, 0, s>>>(dA, mu, m, k, opbits);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_to_integer_res_rows_impl(const T* dA, std::int64_t* dAp, int* mu, int m, int k, cudaStream_t s) {
  kernels::to_integer_rows_k<T><<<m, 128, 0, s>>>(dA, dAp, mu, k);
  CUDA_CHECK(cudaGetLastError());
}

template <int N>
void launch_residues_impl(const std::int64_t* Xp, std::int8_t* Xres, std::size_t count, const CrtConsts<N>& crt, cudaStream_t s) {
  constexpr int threads = 256;
  const auto blocks = static_cast<unsigned>(cuda::ceil_div(count, std::size_t{threads}));
  kernels::residues_k<N><<<blocks, threads, 0, s>>>(Xp, Xres, count, crt);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, int N>
void launch_main_impl(const std::int8_t* Apres, const std::int8_t* Bpres, std::int32_t* Cprod, int m, int n, int k, cudaStream_t s) {
  constexpr int BM = 64, BN = 64;  // must match the defaults of int8_gemm_k
  const dim3 grid(cuda::ceil_div(n, BN), cuda::ceil_div(m, BM));
  kernels::int8_gemm_k<T, N><<<grid, 128, 0, s>>>(Apres, Bpres, Cprod, m, n, k);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, int N>
void launch_crt_impl(const std::int32_t* Cprod, const int* mu, const int* nu, T* C, int m, int n, const CrtConsts<N>& crt, cudaStream_t s) {
  constexpr int threads = 256;
  const auto blocks = static_cast<unsigned>(cuda::ceil_div(std::size_t(m) * n, std::size_t(threads)));
  kernels::crt_k<T, N><<<blocks, threads, 0, s>>>(Cprod, mu, nu, C, m, n, crt);
  CUDA_CHECK(cudaGetLastError());
}

template <std::floating_point T, int M, int N, int K, int numModuli>
requires(numModuli > 1 && numModuli <= 16)
class ogemm {
 public:
  static constexpr auto moduli = [] {
    std::array<int, numModuli> a{};
    std::copy_n(kModuli.begin(), numModuli, a.begin());
    return a;
  }();
  static constexpr int opbits = operand_bits(moduli, K);
  static constexpr u128 P = [] {
    u128 P = 1;
    for (const int p : moduli) P *= static_cast<u128>(p);
    return P;
  }();
  static constexpr auto Pl = [] {
    std::array<u128, numModuli> Pl{};
    for (int i = 0; i < numModuli; ++i) Pl[i] = P/static_cast<u128>(moduli[i]);
    return Pl;
  }();
  static constexpr auto ql = []{
    std::array<int, numModuli> ql{};
    for (std::size_t l = 0; l < numModuli; ++l) {
      const int p = moduli[l];

      int rem = 1;
      for (std::size_t j = 0; j < numModuli; ++j)
        if (j != l) rem = rem * (moduli[j] % p) % p;

      int q = 0;
      for (int cand = 1; cand < p; ++cand)
        if (rem * cand % p == 1) {
          q = cand;
          break;
        }
      if (q == 0) throw std::invalid_argument("moduli must be pairwise coprime");
      ql[l] = q;
    }
    return ql;
  }();
  static constexpr CrtConsts<numModuli> crt = [] {
    CrtConsts<numModuli> c{};
    c.P = P;
    for (int l = 0; l < numModuli; ++l) {
      c.Pl[l] = Pl[l];
      c.p[l] = moduli[l];
      c.q[l] = ql[l];
    }
    return c;
  }();

  explicit ogemm(cuda::stream_ref stream = cuda::stream_ref{cudaStream_t{}}) 
    : stream_(stream), dA_(std::size_t{M} * K), dB_(std::size_t{K} * N), dC_(std::size_t{M} * N),
    dAp_(std::size_t{M} * K), dBp_(std::size_t{K} * N), dApres_(std::size_t{numModuli} * M * K), 
    dBpres_(std::size_t{numModuli} * K * N), dCprod_(std::size_t{numModuli} * M * N), dmu_(M), dnu_(N) {}


  void upload(std::span<const T> hA, std::span<const T> hB) {
    CUDA_CHECK(cudaMemcpyAsync(dA_.data().get(), hA.data(), hA.size_bytes(), cudaMemcpyHostToDevice, stream_.get()));
    CUDA_CHECK(cudaMemcpyAsync(dB_.data().get(), hB.data(), hB.size_bytes(), cudaMemcpyHostToDevice, stream_.get()));
  }

  void compute() {
    launch_row_exponents_impl<T>(dA_.data().get(), dmu_.data().get(), M, K, opbits, stream_.get()); 
    launch_row_exponents_impl<T>(dB_.data().get(), dnu_.data().get(), N, K, opbits, stream_.get());

    launch_to_integer_res_rows_impl<T>(dA_.data().get(), dAp_.data().get(), dmu_.data().get(), M, K, stream_.get());
    launch_to_integer_res_rows_impl<T>(dB_.data().get(), dBp_.data().get(), dnu_.data().get(), N, K, stream_.get());

    launch_residues_impl<numModuli>(dAp_.data().get(), dApres_.data().get(), std::size_t{M} * K, crt, stream_.get());
    launch_residues_impl<numModuli>(dBp_.data().get(), dBpres_.data().get(), std::size_t{N} * K, crt, stream_.get());

    launch_main_impl<T, numModuli>(dApres_.data().get(), dBpres_.data().get(), dCprod_.data().get(), M, N, K, stream_.get());

    launch_crt_impl<T, numModuli>(dCprod_.data().get(), dmu_.data().get(), dnu_.data().get(), dC_.data().get(), M, N, crt, stream_.get());
  }

  void download(std::span<T> hC) {
    CUDA_CHECK(cudaMemcpyAsync(hC.data(), dC_.data().get(), hC.size_bytes(), cudaMemcpyDeviceToHost, stream_.get()));
  }

  void run(std::span<const T> hA, std::span<const T> hB, std::span<T> hC) {
    upload(hA, hB);
    compute();
    download(hC);
    stream_.sync();
  }

  cudaStream_t stream() const { return stream_.get(); }

 private:
  cuda::stream_ref stream_;

  thrust::device_vector<T> dA_; // stored in row major
  thrust::device_vector<T> dB_; // stored in col major
  thrust::device_vector<T> dC_;

  thrust::device_vector<std::int64_t> dAp_;
  thrust::device_vector<std::int64_t> dBp_;

  thrust::device_vector<std::int8_t> dApres_; 
  thrust::device_vector<std::int8_t> dBpres_; 

  thrust::device_vector<std::int32_t> dCprod_;

  thrust::device_vector<int> dmu_;
  thrust::device_vector<int> dnu_;
};
}  // namespace ozaki
