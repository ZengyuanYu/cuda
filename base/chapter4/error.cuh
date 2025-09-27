#pragma once
#include <stdio.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
      printf("CUDA Error:\n");                                                  \
      printf("  File:    %s\n", __FILE__);                                      \
      printf("  Lile:    %d\n", __LINE__);                                      \
      printf("  Error Code:    %d\n", error_code);                              \
      printf("  Error Text:    %s\n", cudaGetErrorString(error_code));          \
      exit(1);                                                                \
    }                                                                          \
  } while (0);
