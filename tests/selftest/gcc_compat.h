// MSVC-only BMI2 intrinsic names, for building the self-test with MinGW-w64 GCC.
// With -mbmi2 GCC emits SHLX/SHRX/SARX/RORX for these expressions.
#pragma once
#if defined(__GNUC__) && !defined(_MSC_VER)
#include <stdint.h>
static inline unsigned int _shlx_u32(unsigned int a, unsigned int n) { return a << (n & 31); }
static inline unsigned long long _shrx_u64(unsigned long long a, unsigned int n) { return a >> (n & 63); }
static inline int _sarx_i32(int a, unsigned int n) { return a >> (n & 31); }
static inline unsigned int _rorx_u32(unsigned int a, const int n) {
  return (a >> (n & 31)) | (a << ((32 - n) & 31));
}
#endif
