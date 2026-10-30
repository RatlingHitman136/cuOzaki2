#pragma once

// CPU implementation of Ozaki scheme II (integer modular technique) for learning.
// Every function below is one step of the pipeline; see CLAUDE.md for the math.
//
//   A (m x k), B (k x n), C (m x n), all row-major.
//   A ~= D_mu * A',  B ~= B' * D_nu,  A', B' integer matrices
//   A'B' is recovered exactly from N small int8 GEMMs (mod p_l) via the CRT
//   C ~= D_mu * (A'B') * D_nu

#include <array>
#include <concepts>
#include <cstdint>
#include <span>
#include <vector>

namespace ozaki {

struct Config {
  int num_moduli = 8;  // N: more moduli => more bits => more accuracy, more GEMMs
};

// Pairwise coprime, all <= 256 so centered residues fit in int8 (p = 256 -> [-128, 127]).
inline constexpr std::array<int, 16> kModuli{256, 255, 253, 251, 247, 241, 239, 233,
                                             229, 227, 223, 211, 199, 197, 193, 191};

// Step 1: exponent e_i with max_k |a_ik| < 2^e_i (0 for an all-zero row). One per row of A.
// Hint: std::frexp.
std::vector<int> row_exponents(std::span<const double> A, int m, int k);
// Same for the columns of B (B is k x n); one exponent per column.
std::vector<int> col_exponents(std::span<const double> B, int k, int n);

// Step 2: bits b per operand such that |A'|, |B'| < 2^b guarantees 2 * k * max|A'| * max|B'| < P,
// where P = product of the moduli. This is what makes the integer product recoverable.
int operand_bits(std::span<const int> moduli, int k);

// Step 3: integer matrices. A'_ik = trunc(a_ik * 2^-mu_i) and B'_kj = trunc(b_kj * 2^-nu_j).
// The result must fit int64, so keep b <= 62.
std::vector<std::int64_t> to_integer_rows(std::span<const double> A, int m, int k,
                                          std::span<const int> mu);
std::vector<std::int64_t> to_integer_cols(std::span<const double> B, int k, int n,
                                          std::span<const int> nu);

// Step 4: x mod p, centered into [-p/2, p/2) and stored as int8. Careful with negative x.
std::vector<std::int8_t> residues(std::span<const std::int64_t> X, int p);

// Step 5: the "tensor core" op on the CPU: C = A * B, int8 x int8 accumulated in int32.
// Row-major, A is m x k, B is k x n.
std::vector<std::int32_t> int8_gemm(std::span<const std::int8_t> A, std::span<const std::int8_t> B,
                                    int m, int n, int k);

// Step 6: Chinese Remainder Theorem. products[l] is the int32 result of GEMM l (modulus
// moduli[l]), so products[l][idx] ≡ (A'B')[idx] (mod p_l) after reducing mod p_l.
// Returns the integer A'B' centered into (-P/2, P/2). P can reach ~2^127, so plain int64 and
// long double are not enough to do this exactly: think about __int128 and reducing early.
std::vector<long double> crt_reconstruct(std::span<const std::vector<std::int32_t>> products,
                                         std::span<const int> moduli);

// Full pipeline (already wired up from the steps above; step 7 = un-scale is done there).
template <std::floating_point T>
void gemm(std::span<const T> A, std::span<const T> B, std::span<T> C, int m, int n, int k,
          const Config& cfg = {});

}  // namespace ozaki
