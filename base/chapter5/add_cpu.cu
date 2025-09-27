#include "error.cuh"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif

const int REPEAT_NUM = 10;
const real a = 1.23;
const real b = 2.34;
const real c = 3.57;

void add(const real *x, const real *y, real *z, const int N);
void check(const real *z, const int N);

int main() {
    const int N = 10000000;
    const int M = sizeof(real) * N;
    real *x = (real*)malloc(M);
    real *y = (real*)malloc(M);
    real *z = (real*)malloc(M);

    for (int n = 0; n < N; ++n) {
        x[n] = a;
        y[n] = b;
    }
    
    float avg_time = 0.0;
    for (int i = 0; i < REPEAT_NUM + 1; i++) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start,  0));
        cudaEventQuery(start);

        add(x, y, z, N);
        
        CHECK(cudaEventRecord(stop,  0));
        CHECK(cudaEventSynchronize(stop));
        float elapesed_time;
        CHECK(cudaEventElapsedTime(&elapesed_time,  start,  stop));
        printf("Time = %g ms\n", elapesed_time);
        if (i > 0) avg_time += elapesed_time;
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        
    }

    printf("AVG Time = %g ms\n", avg_time/REPEAT_NUM);

    check(z, N);

    free(x);
    free(y);
    free(z);
    
    return 0;
}

void add(const real *x, const real *y, real *z, const int N) {
    for (int n = 0; n < N; ++n) {
        z[n] = x[n] + y[n];
    }
}

void check(const real *z, const int N) {
    bool has_error = false;
    for (int n = 0; n < N; ++n) {
        if (fabs(z[n] - c) > EPSILON) {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}