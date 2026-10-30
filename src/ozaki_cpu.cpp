#include "ozaki/ozaki_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ozaki {

std::vector<int> row_exponents(std::span<const double> A, int m, int k) {
  std::vector<int> exps {};

  for (auto row_idx = 0; row_idx < m; row_idx++) {
    double row_max = 0;
    int row_max_exp = 0;
    
    for (auto col_idx = 0; col_idx < k; col_idx++) {
      if (std::fabs(A[row_idx * k + col_idx]) < row_max) continue;

      row_max = A[row_idx * k + col_idx];
      int exp;
      std::frexp(A[row_idx * k + col_idx], &exp);
      row_max_exp = std::max({row_max_exp, exp});
    }

    exps.push_back(row_max_exp);
  }

  return exps;
}

std::vector<int> col_exponents(std::span<const double> B, int k, int n) { 
  std::vector<int> exps {};

  for (auto col_idx = 0; col_idx < n; col_idx++) {
    double col_max = 0;
    int col_max_exp = 0;
    
    for (auto row_idx = 0; row_idx < k; row_idx++) {
      if (std::fabs(B[row_idx * n + col_idx]) < col_max) continue;

      col_max = B[row_idx * n + col_idx];
      int exp;
      std::frexp(B[row_idx * n + col_idx], &exp);
      col_max_exp = std::max({col_max_exp, exp});
    }

    exps.push_back(col_max_exp);
  }

  return exps;
}

int operand_bits(std::span<const int> moduli, int k) { 
  double res = 0;
  for (const auto m : moduli) {
    res += std::log2(m);
  }

  return std::floor((res - std::log2(k) - 1) / 2);
}

std::vector<std::int64_t> to_integer_rows(std::span<const double> A, int m, int k, std::span<const int> mu) {
  std::vector<std::int64_t> res(m * k);
  for (auto i = 0; i < m; i++) {
    for (auto l = 0; l < k; l++) {
      res[i*k + l] = static_cast<int64_t>(std::trunc(std::ldexp(A[i*k + l], -mu[i])));
    }
  }

  return res;
}

std::vector<std::int64_t> to_integer_cols(std::span<const double> B, int k, int n, std::span<const int> nu) {
  std::vector<std::int64_t> res(k * n);
  for (auto l = 0; l < k; l++) {
    for (auto j = 0; j < n; j++) {

      res[l*n + j] = static_cast<int64_t>(std::trunc(std::ldexp(B[l*n + j], -nu[j])));
    }
  }

  return res;
}

std::vector<std::int8_t> residues(std::span<const std::int64_t> X, int p) { 
  std::vector<std::int8_t> res(X.size());
  for (std::size_t i = 0; i < X.size(); i++) {
    auto tmp = X[i] % p;
    if (tmp < 0)
      tmp += p;
    if (tmp >= (p + 1)/2)
      tmp -= p;
  
    res[i] = static_cast<int8_t>(tmp);
  }

  return res;
}

std::vector<std::int32_t> int8_gemm(std::span<const std::int8_t> A, std::span<const std::int8_t> B, int m, int n, int k) {
  std::vector<std::int32_t> C = std::vector<int32_t>(m * n);

  for (auto i = 0; i < m; ++i) {
    for (auto j = 0; j < n; j++) {
      std::int32_t acc = 0;
      for (auto l = 0; l < k; l++) {
        acc += A[i * k + l] * B[l * n + j];
      }
      C[i * n + j] = acc;
    }
  }

  return C;
}

std::vector<long double> crt_reconstruct(std::span<const std::vector<std::int32_t>> products, std::span<const int> moduli) {
  __extension__ typedef unsigned __int128 u128;

  const std::size_t N = moduli.size();
  if (products.size() != N) throw std::invalid_argument("one product per modulus");
  for (const auto& c : products)
    if (c.size() != products[0].size()) throw std::invalid_argument("product sizes differ");

  // Precompute P, P_l = P / p_l and q_l = P_l^-1 mod p_l once, outside the per-element loop.
  u128 P = 1;
  for (const int p : moduli) P *= static_cast<u128>(p);

  std::vector<u128> Pl(N);
  std::vector<int> ql(N);
  for (std::size_t l = 0; l < N; ++l) {
    const int p = moduli[l];
    Pl[l] = P / static_cast<u128>(p);

    int rem = 1;  // P_l mod p, built from the other moduli with small-int arithmetic
    for (std::size_t j = 0; j < N; ++j)
      if (j != l) rem = rem * (moduli[j] % p) % p;

    int q = 0;  // brute-force modular inverse; cheap because p <= 256
    for (int cand = 1; cand < p; ++cand)
      if (rem * cand % p == 1) {
        q = cand;
        break;
      }
    if (q == 0) throw std::invalid_argument("moduli must be pairwise coprime");
    ql[l] = q;
  }

  const std::size_t count = products[0].size();
  std::vector<long double> out(count);
  for (std::size_t idx = 0; idx < count; ++idx) {
    u128 s = 0;
    for (std::size_t l = 0; l < N; ++l) {
      const int p = moduli[l];
      int c = products[l][idx] % p;  // C++ % keeps the sign of the dividend ...
      if (c < 0) c += p;             // ... so bring it into [0, p)
      const int t = c * ql[l] % p;
      s += static_cast<u128>(t) * Pl[l];  // t * P_l < P
      if (s >= P) s -= P;                 // reduce early: keeps s + term < 2P < 2^127
    }
    // Center into (-P/2, P/2): values in the upper half represent negative integers.
    out[idx] = s > P / 2 ? -static_cast<long double>(P - s) : static_cast<long double>(s);
  }
  return out;
}

template <std::floating_point T>
void gemm(std::span<const T> A, std::span<const T> B, std::span<T> C, int m, int n, int k,
          const Config& cfg) {
  const std::span<const int> moduli = std::span<const int>(kModuli).first(cfg.num_moduli);

  const std::vector<double> Ad(A.begin(), A.end());  // float -> double is exact
  const std::vector<double> Bd(B.begin(), B.end());

  // Scales: 2^mu_i = 2^e_i / 2^b, so that the largest |A'_ik| in row i is < 2^b.
  const int b = operand_bits(moduli, k);
  std::vector<int> mu = row_exponents(Ad, m, k);
  std::vector<int> nu = col_exponents(Bd, k, n);
  for (int& e : mu) e -= b;
  for (int& e : nu) e -= b;

  const auto Ap = to_integer_rows(Ad, m, k, mu);
  const auto Bp = to_integer_cols(Bd, k, n, nu);

  // N independent exact small-modulus GEMMs.
  std::vector<std::vector<std::int32_t>> products;
  products.reserve(moduli.size());
  for (const int p : moduli)
    products.push_back(int8_gemm(residues(Ap, p), residues(Bp, p), m, n, k));

  const std::vector<long double> Cp = crt_reconstruct(products, moduli);

  // Step 7: C = D_mu * C' * D_nu, exact because the scales are powers of two.
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      const std::size_t idx = static_cast<std::size_t>(i) * n + j;
      C[idx] = static_cast<T>(std::ldexp(Cp[idx], mu[i] + nu[j]));
    }
}

template void gemm<float>(std::span<const float>, std::span<const float>, std::span<float>, int,
                          int, int, const Config&);
template void gemm<double>(std::span<const double>, std::span<const double>, std::span<double>,
                           int, int, int, const Config&);

}  // namespace ozaki::scheme2
