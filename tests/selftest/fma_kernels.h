#pragma once
#include <stdint.h>
typedef void (*fma_kernel_fn)(const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* out);
struct FmaKernel { const char* name; fma_kernel_fn fn; };
extern "C" int get_fma_kernels(const FmaKernel** list);    // native FMA/BMI1 (fma_kernels.cpp)
extern "C" int get_fma_reference(const FmaKernel** list);  // plain AVX reference (fma_reference.cpp)
