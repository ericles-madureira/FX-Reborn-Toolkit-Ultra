// Micro-benchmark for converted code: hot loops made of AVX2 / FMA instructions.
// Build for a CPU WITH AVX2+FMA (-O2 -mavx2 -mfma), convert with fxkit, run the
// converted exe on the AVX-only machine and compare nanoseconds per iteration.
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <windows.h>

static volatile float vf[64];
static volatile int vi[64];
static double now() { LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f); return (double)c.QuadPart / f.QuadPart; }
#define ITER 20000000

__attribute__((noinline)) static float fma_accumulate(int n) {
  __m256 x = _mm256_set1_ps(vf[0]), y = _mm256_set1_ps(vf[1]);
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
  for (int i = 0; i < n; ++i) {
    a0 = _mm256_fmadd_ps(x, y, a0); a1 = _mm256_fmadd_ps(y, x, a1);
    a2 = _mm256_fnmadd_ps(x, y, a2); a3 = _mm256_fmsub_ps(x, a3, y);
    __asm__ volatile("" : "+x"(x), "+x"(y));
  }
  __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
  return _mm256_cvtss_f32(s);
}
__attribute__((noinline)) static float fma_scalar(int n) {
  __m128 x = _mm_set_ss(vf[2]), y = _mm_set_ss(vf[3]), a = _mm_set_ss(0), b = a;
  for (int i = 0; i < n; ++i) {
    a = _mm_fmadd_ss(x, y, a); b = _mm_fnmadd_ss(y, x, b);
    __asm__ volatile("" : "+x"(x), "+x"(y));
  }
  return _mm_cvtss_f32(_mm_add_ss(a, b));
}
__attribute__((noinline)) static int avx2_int(int n) {
  __m256i x = _mm256_set1_epi32(vi[0]), y = _mm256_set1_epi32(vi[1]), acc = _mm256_setzero_si256();
  for (int i = 0; i < n; ++i) {
    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(x, y));
    acc = _mm256_xor_si256(acc, _mm256_slli_epi32(acc, 3));
    __asm__ volatile("" : "+x"(x), "+x"(y));
  }
  return _mm256_extract_epi32(acc, 0) ^ _mm256_extract_epi32(acc, 7);
}

int main() {
  for (int i = 0; i < 64; ++i) { vf[i] = 1.0f + i * 1e-7f; vi[i] = i * 2654435761u; }
  struct { const char* name; double (*run)(); } tests[] = {
    {"fma_accumulate (4x ymm, 231/213)", [] { double t = now(); volatile float r = fma_accumulate(ITER); (void)r; return now() - t; }},
    {"fma_scalar (ss 231)", [] { double t = now(); volatile float r = fma_scalar(ITER); (void)r; return now() - t; }},
    {"avx2_int (vpmulld/vpaddd/vpslld)", [] { double t = now(); volatile int r = avx2_int(ITER); (void)r; return now() - t; }},
  };
  for (auto& t : tests) {
    double best = 1e9;
    for (int rep = 0; rep < 5; ++rep) { double s = t.run(); if (s < best) best = s; }
    printf("%-36s %7.2f ns/iter\n", t.name, best * 1e9 / ITER);
  }
  return 0;
}
