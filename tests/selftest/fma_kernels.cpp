// FMA3 / BMI1 / LZCNT / MOVBE kernels, compiled for a CPU that HAS these
// features (-mavx2 -mfma -mbmi -mlzcnt -mmovbe). After conversion they must
// match fma_reference.cpp (compiled for plain AVX) bit for bit.
// Emulation contract for FMA: a*b is rounded, then the add/sub is rounded.
#include "fma_kernels.h"
#include <immintrin.h>
#include <string.h>

#define PS(p) _mm256_loadu_ps((const float*)(p))
#define PD(p) _mm256_loadu_pd((const double*)(p))
#define PS4(p) _mm_loadu_ps((const float*)(p))
#define PD2(p) _mm_loadu_pd((const double*)(p))
#define K(name) static void name(const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* o)

K(fmadd_ps)    { _mm256_storeu_ps((float*)o, _mm256_fmadd_ps(PS(a), PS(b), PS(c))); }
K(fmsub_ps)    { _mm256_storeu_ps((float*)o, _mm256_fmsub_ps(PS(a), PS(b), PS(c))); }
K(fnmadd_ps)   { _mm256_storeu_ps((float*)o, _mm256_fnmadd_ps(PS(a), PS(b), PS(c))); }
K(fnmsub_ps)   { _mm256_storeu_ps((float*)o, _mm256_fnmsub_ps(PS(a), PS(b), PS(c))); }
K(fmaddsub_ps) { _mm256_storeu_ps((float*)o, _mm256_fmaddsub_ps(PS(a), PS(b), PS(c))); }
K(fmsubadd_ps) { _mm256_storeu_ps((float*)o, _mm256_fmsubadd_ps(PS(a), PS(b), PS(c))); }
K(fmadd_pd)    { _mm256_storeu_pd((double*)o, _mm256_fmadd_pd(PD(a), PD(b), PD(c))); }
K(fnmadd_pd)   { _mm256_storeu_pd((double*)o, _mm256_fnmadd_pd(PD(a), PD(b), PD(c))); }
K(fmsubadd_pd) { _mm256_storeu_pd((double*)o, _mm256_fmsubadd_pd(PD(a), PD(b), PD(c))); }
K(fmadd_ps128) { memset(o, 0, 32); _mm_storeu_ps((float*)o, _mm_fmadd_ps(PS4(a), PS4(b), PS4(c))); }
K(fnmsub_pd128){ memset(o, 0, 32); _mm_storeu_pd((double*)o, _mm_fnmsub_pd(PD2(a), PD2(b), PD2(c))); }
K(fmadd_ss)    { memset(o, 0, 32); _mm_storeu_ps((float*)o, _mm_fmadd_ss(PS4(a), PS4(b), PS4(c))); }
K(fmsub_ss)    { memset(o, 0, 32); _mm_storeu_ps((float*)o, _mm_fmsub_ss(PS4(a), PS4(b), PS4(c))); }
K(fnmadd_ss)   { memset(o, 0, 32); _mm_storeu_ps((float*)o, _mm_fnmadd_ss(PS4(a), PS4(b), PS4(c))); }
K(fnmsub_sd)   { memset(o, 0, 32); _mm_storeu_pd((double*)o, _mm_fnmsub_sd(PD2(a), PD2(b), PD2(c))); }
K(fmadd_sd)    { memset(o, 0, 32); _mm_storeu_pd((double*)o, _mm_fmadd_sd(PD2(a), PD2(b), PD2(c))); }

// Accumulate loop: compilers emit the 231 form (acc += x*y), the most common FMA in games.
K(dot8x4_ps) {
  __m256 acc = PS(c);
  for (int i = 0; i < 4; ++i) acc = _mm256_fmadd_ps(PS(a + 32 * i), PS(b + 32 * i), acc);
  _mm256_storeu_ps((float*)o, acc);
}
// Horner polynomial: the 213 form (r = r*x + k).
K(horner_ps) {
  __m256 x = PS(a), r = PS(b);
  for (int i = 0; i < 4; ++i) r = _mm256_fmadd_ps(r, x, PS(c + 32 * i));
  _mm256_storeu_ps((float*)o, r);
}
// Scalar accumulate (vfmadd231ss) and a memory multiplicand.
K(dot_ss) {
  float acc = ((const float*)c)[0];
  for (int i = 0; i < 16; ++i)
    acc = _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(((const float*)a)[i]), _mm_set_ss(((const float*)b)[i]), _mm_set_ss(acc)));
  memset(o, 0, 32); memcpy(o, &acc, 4);
}
K(bmi1_mix) {
  uint64_t x, y; uint32_t u, v;
  memcpy(&x, a, 8); memcpy(&y, b, 8); memcpy(&u, a + 8, 4); memcpy(&v, b + 8, 4);
  uint64_t r[4];
  r[0] = _andn_u64(x, y) ^ (_blsr_u64(x) << 1) ^ _blsi_u64(y) ^ _blsmsk_u64(x | 1);
  r[1] = (uint64_t)_andn_u32(u, v) | ((uint64_t)_bextr_u32(u, v & 31, (v >> 8) & 31) << 32);
  r[2] = (uint64_t)_tzcnt_u64(x & (y | 0x8000000000000000ull)) | ((uint64_t)_lzcnt_u32(u >> (v & 31)) << 8)
       | ((uint64_t)_lzcnt_u64(y >> (u & 63)) << 16) | ((uint64_t)_tzcnt_u32(v) << 24);
  r[3] = (uint64_t)__builtin_bswap32(*(const uint32_t*)(c + 4)) | ((uint64_t)__builtin_bswap16(*(const uint16_t*)(c + 10)) << 32);
  memcpy(o, r, 32);
}

static const FmaKernel table[] = {
  {"fmadd_ps", fmadd_ps}, {"fmsub_ps", fmsub_ps}, {"fnmadd_ps", fnmadd_ps}, {"fnmsub_ps", fnmsub_ps},
  {"fmaddsub_ps", fmaddsub_ps}, {"fmsubadd_ps", fmsubadd_ps}, {"fmadd_pd", fmadd_pd}, {"fnmadd_pd", fnmadd_pd},
  {"fmsubadd_pd", fmsubadd_pd}, {"fmadd_ps128", fmadd_ps128}, {"fnmsub_pd128", fnmsub_pd128},
  {"fmadd_ss", fmadd_ss}, {"fmsub_ss", fmsub_ss}, {"fnmadd_ss", fnmadd_ss}, {"fnmsub_sd", fnmsub_sd},
  {"fmadd_sd", fmadd_sd}, {"dot8x4_ps", dot8x4_ps}, {"horner_ps", horner_ps}, {"dot_ss", dot_ss},
  {"bmi1_mix", bmi1_mix},
};
extern "C" int get_fma_kernels(const FmaKernel** list) { *list = table; return sizeof(table) / sizeof(table[0]); }
