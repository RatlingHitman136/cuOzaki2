#include <cublas_v2.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <source_location>
#include <vector>

#include "ozaki/ozaki.cuh"

namespace {

constexpr int M = 4096, N = 4096, K = 4096;
constexpr int kSamples = 512;

struct Event {
  cudaEvent_t e{};
  Event() { CUDA_CHECK(cudaEventCreate(&e)); }
  ~Event() { cudaEventDestroy(e); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

void cublas_check(cublasStatus_t st, std::source_location loc = std::source_location::current()) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuBLAS error %d at %s:%u\n", static_cast<int>(st), loc.file_name(), loc.line());
    std::exit(EXIT_FAILURE);
  }
}

struct CublasHandle {
  cublasHandle_t h{};
  CublasHandle() { cublas_check(cublasCreate(&h)); }
  ~CublasHandle() { cublasDestroy(h); }
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
};


template <class F>
double time_ms(cudaStream_t s, int warmup, int reps, F&& launch) {
  for (int i = 0; i < warmup; ++i) launch();
  Event beg, end;
  CUDA_CHECK(cudaEventRecord(beg.e, s));
  for (int i = 0; i < reps; ++i) launch();
  CUDA_CHECK(cudaEventRecord(end.e, s));
  CUDA_CHECK(cudaEventSynchronize(end.e));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, beg.e, end.e));
  return ms / reps;
}


struct DD {
  double hi, lo;
};

DD two_sum(double a, double b) {
  const double s = a + b;
  const double bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}

void dd_fma(DD& acc, double a, double b) {
  const double p = a * b;
  const double pe = std::fma(a, b, -p);  // a * b == p + pe exactly
  const DD s = two_sum(acc.hi, p);
  const double lo = s.lo + (acc.lo + pe);
  acc.hi = s.hi + lo;
  acc.lo = lo - (acc.hi - s.hi);
}

struct Sample {
  int i, j;
  long double ref;
  double abs_sum; 
};

std::vector<Sample> make_reference(const std::vector<double>& hA, const std::vector<double>& hB) {
  std::mt19937_64 rng(1234);
  std::uniform_int_distribution<int> pick_i(0, M - 1), pick_j(0, N - 1);
  std::vector<Sample> out(kSamples);
  for (auto& s : out) {
    s.i = pick_i(rng);
    s.j = pick_j(rng);
    const double* a = &hA[std::size_t{static_cast<std::size_t>(s.i)} * K];  // row i of A
    const double* b = &hB[std::size_t{static_cast<std::size_t>(s.j)} * K];  // column j of B (col-major)
    DD acc{0.0, 0.0};
    double abs_sum = 0.0;
    for (int k = 0; k < K; ++k) {
      dd_fma(acc, a[k], b[k]);
      abs_sum += std::fabs(a[k] * b[k]);
    }
    s.ref = static_cast<long double>(acc.hi) + acc.lo;
    s.abs_sum = abs_sum;
  }
  return out;
}

struct Errors {
  double max_scaled = 0.0, mean_scaled = 0.0, max_ulp = 0.0;
};

Errors measure(const std::vector<double>& C, const std::vector<Sample>& ref) {
  constexpr double u = 0x1p-53;
  Errors e;
  for (const auto& s : ref) {
    const double got = C[static_cast<std::size_t>(s.i) * N + s.j];
    const long double diff = std::fabs(static_cast<long double>(got) - s.ref);
    const double scaled = static_cast<double>(diff / s.abs_sum) / u;
    const double r = std::fabs(static_cast<double>(s.ref));
    const double ulp = std::nextafter(r, std::numeric_limits<double>::infinity()) - r;
    e.max_scaled = std::max(e.max_scaled, scaled);
    e.mean_scaled += scaled / static_cast<double>(ref.size());
    e.max_ulp = std::max(e.max_ulp, static_cast<double>(diff / ulp));
  }
  return e;
}

struct Result {
  double ms;
  Errors err;
};

void print_row(const char* name, int moduli, int bits, const Result& r, double base_ms) {
  const double tflops = 2.0 * M * N * K / (r.ms * 1e-3) / 1e12;
  char label[48];
  if (moduli > 0)
    std::snprintf(label, sizeof label, "%s, N=%d (b=%d)", name, moduli, bits);
  else
    std::snprintf(label, sizeof label, "%s", name);
  std::printf("%-26s %9.2f %9.3f %8.2fx %14.2f %10.2f %12.1f\n", label, r.ms, tflops, base_ms / r.ms,
              r.err.max_scaled, r.err.mean_scaled, r.err.max_ulp);
}

Result bench_cublas(const std::vector<double>& hA, const std::vector<double>& hB,
                    const std::vector<Sample>& ref, std::vector<double>& hC) {
  thrust::device_vector<double> dA = hA, dB = hB, dC(std::size_t{M} * N);
  CublasHandle handle;
  const double alpha = 1.0, beta = 0.0;
  auto gemm = [&] {
    cublas_check(cublasDgemm(handle.h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB.data().get(), K,
                             dA.data().get(), K, &beta, dC.data().get(), N));
  };
  const double ms = time_ms(nullptr, 2, 5, gemm);
  CUDA_CHECK(cudaMemcpy(hC.data(), dC.data().get(), hC.size() * sizeof(double), cudaMemcpyDeviceToHost));
  return {ms, measure(hC, ref)};
}

template <int NumModuli>
Result bench_ozaki(const std::vector<double>& hA, const std::vector<double>& hB,
                   const std::vector<Sample>& ref, std::vector<double>& hC, int& bits) {
  ozaki::ogemm<double, M, N, K, NumModuli> g;
  bits = decltype(g)::opbits;
  g.upload(hA, hB);
  const double ms = time_ms(g.stream(), 2, 5, [&] { g.compute(); });
  g.download(hC);
  CUDA_CHECK(cudaStreamSynchronize(g.stream()));
  return {ms, measure(hC, ref)};
}

}  // namespace

int main() {
#ifndef NDEBUG
  std::fputs("warning: not a Release build, the timings are meaningless\n", stderr);
#endif
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\n", prop.name);
  std::printf("DGEMM %d x %d x %d, entries ~ U(-1, 1), accuracy from %d sampled entries vs a double-double reference\n",
              M, N, K, kSamples);
  std::puts("TFLOP/s counts 2*M*N*K flops for both; b = operand bits; errors in units of u = 2^-53 relative to sum|a*b|\n");

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> hA(std::size_t{M} * K), hB(std::size_t{K} * N), hC(std::size_t{M} * N);
  for (auto& x : hA) x = dist(rng);
  for (auto& x : hB) x = dist(rng);  // hB is column-major: element (k, j) at hB[j * K + k]

  const std::vector<Sample> ref = make_reference(hA, hB);

  std::printf("%-26s %9s %9s %9s %14s %10s %12s\n", "", "ms", "TFLOP/s", "speedup", "max err [u]", "mean err", "max err [ulp]");
  const Result base = bench_cublas(hA, hB, ref, hC);
  print_row("cuBLAS DGEMM", 0, 0, base, base.ms);

  int bits = 0;
  Result r = bench_ozaki<14>(hA, hB, ref, hC, bits);
  print_row("Ozaki II (int8)", 14, bits, r, base.ms);
  r = bench_ozaki<15>(hA, hB, ref, hC, bits);
  print_row("Ozaki II (int8)", 15, bits, r, base.ms);
  r = bench_ozaki<16>(hA, hB, ref, hC, bits);
  print_row("Ozaki II (int8)", 16, bits, r, base.ms);
  return 0;
}
