// Runs every FMA/BMI1 kernel on random finite inputs (plus exact cancellations,
// zeros and bit-pattern edge cases) and compares with the plain-AVX reference.
#include "fma_kernels.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
static uint64_t s = 0xD1B54A32D192ED03ull;
static uint64_t rnd() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static float rf() { int e = (int)(rnd() % 41) - 20; float m = (float)(rnd() % 1000000) / 1000000.0f; return (rnd() & 1 ? -m : m) * ldexpf(1.0f, e); }
static double rd() { int e = (int)(rnd() % 121) - 60; double m = (double)(rnd() % 1000000007ull) / 1e9; return (rnd() & 1 ? -m : m) * ldexp(1.0, e); }
int main() {
  const FmaKernel *k, *r;
  int n = get_fma_kernels(&k), m = get_fma_reference(&r), failed = 0;
  if (n != m) { printf("table mismatch\n"); return 2; }
  for (int t = 0; t < n; ++t) {
    int bad = 0;
    for (int it = 0; it < 4000; ++it) {
      uint8_t a[128], b[128], c[128], o1[32], o2[32];
      bool dbl = strstr(k[t].name, "_pd") || strstr(k[t].name, "_sd");
      for (int i = 0; i < 128; i += 8) {
        if (it % 5 == 4) { uint64_t v = rnd(); memcpy(a + i, &v, 8); v = rnd(); memcpy(b + i, &v, 8); v = rnd(); memcpy(c + i, &v, 8); continue; }
        if (dbl) { double x = rd(), y = rd(), z = rd(); memcpy(a + i, &x, 8); memcpy(b + i, &y, 8); memcpy(c + i, &z, 8); }
        else for (int j = 0; j < 8; j += 4) { float x = rf(), y = rf(), z = rf(); memcpy(a + i + j, &x, 4); memcpy(b + i + j, &y, 4); memcpy(c + i + j, &z, 4); }
      }
      if (it % 7 == 3) {  // exact products and exact cancellation: sign of zero results
        for (int i = 0; i < 32; i += (dbl ? 8 : 4)) {
          if (dbl) { double x = (double)(int)(rnd() % 64) - 32, y = (double)(int)(rnd() % 64) - 32, z = (rnd() & 1) ? x * y : -(x * y);
                     memcpy(a + i, &x, 8); memcpy(b + i, &y, 8); memcpy(c + i, &z, 8); }
          else { float x = (float)(int)(rnd() % 64) - 32, y = (float)(int)(rnd() % 64) - 32, z = (rnd() & 1) ? x * y : -(x * y);
                 memcpy(a + i, &x, 4); memcpy(b + i, &y, 4); memcpy(c + i, &z, 4); }
        }
      }
      bool ints = strcmp(k[t].name, "bmi1_mix") == 0;
      if (!ints && it % 5 == 4) continue;  // raw bit patterns only for the integer kernel
      if (ints && it % 4 == 1) { memset(a, 0, 16); }
      memset(o1, 0xCC, 32); memset(o2, 0xCC, 32);
      k[t].fn(a, b, c, o1); r[t].fn(a, b, c, o2);
      if (memcmp(o1, o2, 32)) {
        if (!bad++) {
          printf("FAIL %-12s first mismatch at iteration %d\n  got ", k[t].name, it);
          for (int i = 0; i < 32; ++i) printf("%02x", o1[i]);
          printf("\n  ref ");
          for (int i = 0; i < 32; ++i) printf("%02x", o2[i]);
          printf("\n");
        }
      }
    }
    if (bad) ++failed; else printf("ok   %s\n", k[t].name);
  }
  if (failed) printf("%d of %d kernels FAILED\n", failed, n); else printf("ALL %d FMA/BMI1 KERNELS PASS\n", n);
  return failed ? 1 : 0;
}
