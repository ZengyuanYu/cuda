#include "error.cuh"
#include <cuda_device_runtime_api.h>
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void __global__ add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main() {
    const int N = 10000000 + 1;
    const int M = sizeof(double) * N;
    double *host_x = (double*)malloc(M);
    double *host_y = (double*)malloc(M);
    double *host_z = (double*)malloc(M);

    for (int n = 0; n < N; ++n) {
        host_x[n] = a;
        host_y[n] = b;
    }

    double *device_x, *device_y, *device_z;
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

    free(host_x);
    free(host_y);
    free(host_z);
    
    CHECK(cudaFree(device_x));
    CHECK(cudaFree(device_y));
    CHECK(cudaFree(device_z));

    return 0;
}

void __global__ add(const double *x, const double *y, double *z, const int N) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n >= N) return;
    z[n] = x[n] + y[n];
}

void check(const double *z, const int N) {
    bool has_error = false;
    for (int n = 0; n < N; ++n) {
        if (fabs(z[n] - c) > EPSILON) {
            // printf("%d-%f", n, z[n]);
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}