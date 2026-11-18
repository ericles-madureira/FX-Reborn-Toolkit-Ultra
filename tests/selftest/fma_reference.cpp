// Scalar reference for fma_kernels.cpp. Build with -mavx -ffp-contract=off
// (or MSVC /arch:AVX /fp:precise): every product and sum is rounded separately.
#include "fma_kernels.h"
#include <string.h>

#define K(name) static void name(const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* o)
template <class T> static T at(const uint8_t* p, int i) { T v; memcpy(&v, p + i * sizeof(T), sizeof(T)); return v; }
template <class T> static void put(uint8_t* p, int i, T v) { memcpy(p + i * sizeof(T), &v, sizeof(T)); }

// mode: 0 add, 1 sub, 2 nadd (-p+c), 3 nsub (-p-c), 4 addsub (even sub), 5 subadd (even add)
template <class T> static T op(T x, T y, T z, int mode, int lane) {
  volatile T p = x * y;
  T q = p;
  switch (mode) {
    case 0: return q + z;
    case 1: return q - z;
    case 2: return z - q;
    case 3: return -q - z;
    case 4: return (lane & 1) ? q + z : q - z;
    default: return (lane & 1) ? q - z : q + z;
  }
}
template <class T> static void packed(const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* o, int n, int mode) {
  memset(o, 0, 32);
  for (int i = 0; i < n; ++i) put<T>(o, i, op<T>(at<T>(a, i), at<T>(b, i), at<T>(c, i), mode, i));
}
template <class T> static void scalar(const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* o, int mode) {
  memset(o, 0, 32);
  memcpy(o, a, 16);
  put<T>(o, 0, op<T>(at<T>(a, 0), at<T>(b, 0), at<T>(c, 0), mode, 0));
}

K(fmadd_ps)    { packed<float>(a, b, c, o, 8, 0); }
K(fmsub_ps)    { packed<float>(a, b, c, o, 8, 1); }
K(fnmadd_ps)   { packed<float>(a, b, c, o, 8, 2); }
K(fnmsub_ps)   { packed<float>(a, b, c, o, 8, 3); }
K(fmaddsub_ps) { packed<float>(a, b, c, o, 8, 4); }
K(fmsubadd_ps) { packed<float>(a, b, c, o, 8, 5); }
K(fmadd_pd)    { packed<double>(a, b, c, o, 4, 0); }
K(fnmadd_pd)   { packed<double>(a, b, c, o, 4, 2); }
K(fmsubadd_pd) { packed<double>(a, b, c, o, 4, 5); }
K(fmadd_ps128) { packed<float>(a, b, c, o, 4, 0); }
K(fnmsub_pd128){ packed<double>(a, b, c, o, 2, 3); }
K(fmadd_ss)    { scalar<float>(a, b, c, o, 0); }
K(fmsub_ss)    { scalar<float>(a, b, c, o, 1); }
K(fnmadd_ss)   { scalar<float>(a, b, c, o, 2); }
K(fnmsub_sd)   { scalar<double>(a, b, c, o, 3); }
K(fmadd_sd)    { scalar<double>(a, b, c, o, 0); }
K(dot8x4_ps) {
  memset(o, 0, 32);
  for (int l = 0; l < 8; ++l) {
    float acc = at<float>(c, l);
    for (int i = 0; i < 4; ++i) acc = op<float>(at<float>(a + 32 * i, l), at<float>(b + 32 * i, l), acc, 0, 0);
    put<float>(o, l, acc);
  }
}
K(horner_ps) {
  memset(o, 0, 32);
  for (int l = 0; l < 8; ++l) {
    float x = at<float>(a, l), r = at<float>(b, l);
    for (int i = 0; i < 4; ++i) r = op<float>(r, x, at<float>(c + 32 * i, l), 0, 0);
    put<float>(o, l, r);
  }
}
K(dot_ss) {
  float acc = at<float>(c, 0);
  for (int i = 0; i < 16; ++i) acc = op<float>(at<float>(a, i), at<float>(b, i), acc, 0, 0);
  memset(o, 0, 32); memcpy(o, &acc, 4);
}

static uint64_t tz64(uint64_t v) { if (!v) return 64; uint64_t n = 0; while (!(v & 1)) { v >>= 1; ++n; } return n; }
static uint64_t lz64(uint64_t v) { uint64_t n = 0; for (int i = 63; i >= 0 && !((v >> i) & 1); --i) ++n; return n; }
static uint32_t bextr32(uint32_t s, uint32_t start, uint32_t len) {
  if (start >= 32) return 0;
  uint64_t v = (uint64_t)s >> start;
  return len >= 32 ? (uint32_t)v : (uint32_t)(v & ((1ull << len) - 1));
}
K(bmi1_mix) {
  uint64_t x, y; uint32_t u, v;
  memcpy(&x, a, 8); memcpy(&y, b, 8); memcpy(&u, a + 8, 4); memcpy(&v, b + 8, 4);
  uint64_t z = x | 1;
  uint64_t r[4];
  r[0] = (~x & y) ^ ((x & (x - 1)) << 1) ^ (y & (0 - y)) ^ (z ^ (z - 1));
  r[1] = (uint64_t)(~u & v) | ((uint64_t)bextr32(u, v & 31, (v >> 8) & 31) << 32);
  uint32_t w = u >> (v & 31);
  r[2] = tz64(x & (y | 0x8000000000000000ull)) | ((uint64_t)(w ? lz64(w) - 32 : 32) << 8)
       | ((uint64_t)lz64(y >> (u & 63)) << 16) | ((uint64_t)(v ? tz64(v) : 32) << 24);
  uint32_t m; uint16_t h; memcpy(&m, c + 4, 4); memcpy(&h, c + 10, 2);
  m = (m >> 24) | ((m >> 8) & 0xFF00) | ((m << 8) & 0xFF0000) | (m << 24);
  h = (uint16_t)((h >> 8) | (h << 8));
  r[3] = (uint64_t)m | ((uint64_t)h << 32);
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
extern "C" int get_fma_reference(const FmaKernel** list) { *list = table; return sizeof(table) / sizeof(table[0]); }
