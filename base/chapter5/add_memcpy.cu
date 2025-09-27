#include "error.cuh"
#include <cuda_device_runtime_api.h>
#include <math.h>
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

void __global__ add(const real *x, const real *y, real *z, const int N);
void check(const real *z, const int N);

int main() {
    const int N = 10000000;
    const int M = sizeof(real) * N;
    real *host_x = (real*)malloc(M);
    real *host_y = (real*)malloc(M);
    real *host_z = (real*)malloc(M);

    for (int n = 0; n < N; ++n) {
        host_x[n] = a;
        host_y[n] = b;
    }

    real *device_x, *device_y, *device_z;
        
    float avg_time = 0.0;
    for (int i = 0; i < REPEAT_NUM + 1; i++) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start,  0));
        cudaEventQuery(start);

        // cudaMalloc(void** adress, size_t size);
        CHECK(cudaMalloc((void **)&device_x, M));
        CHECK(cudaMalloc((void **)&device_y, M));
        CHECK(cudaMalloc((void **)&device_z, M));

        CHECK(cudaMemcpy(device_x, host_x, M, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(device_y, host_y, M, cudaMemcpyHostToDevice));

        // const int block_size = 1280;
        const int block_size = 128;
        const int grid_size  = (N - 1) / block_size + 1;

        add<<<grid_size, block_size>>>(device_x, device_y, device_z, N); 
     
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(host_z, device_z, M, cudaMemcpyDeviceToHost));
        check(host_z, N);
        
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

    free(host_x);
    free(host_y);
    free(host_z);
    
    CHECK(cudaFree(device_x));
    CHECK(cudaFree(device_y));
    CHECK(cudaFree(device_z));

    return 0;
}

void __global__ add(const real *x, const real *y, real *z, const int N) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n >= N) return;
    z[n] = x[n] + y[n];
}

void check(const real *z, const int N) {
    bool has_error = false;
    for (int n = 0; n < N; ++n) {
        if (fabs(z[n] - c) > EPSILON) {
            // printf("%d-%f", n, z[n]);
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}