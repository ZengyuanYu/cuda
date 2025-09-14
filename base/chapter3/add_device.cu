#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void   __global__ add(const double *x, const double *y, double *z, const int N);
double __device__ add_device(const double x, const double y);

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
    cudaMalloc((void **)&device_x, M);
    cudaMalloc((void **)&device_y, M);
    cudaMalloc((void **)&device_z, M);

    cudaMemcpy(device_x, host_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, host_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size  = (N - 1) / block_size + 1;
    add<<<grid_size, block_size>>>(device_x, device_y, device_z, N); 
    cudaMemcpy(host_z, device_z, M, cudaMemcpyDeviceToHost);
    check(host_z, N);

    free(host_x);
    free(host_y);
    free(host_z);
    
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);

    return 0;
}

void __global__ add(const double *x, const double *y, double *z, const int N) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n >= N) return;
    z[n] = add_device(x[n], y[n]);
}

double __device__ add_device(const double x, const double y) {
    return (x + y);
}

void check(const double *z, const int N) {
    bool has_error = false;
    for (int n = 0; n < N; ++n) {
        if (fabs(z[n] - c) > EPSILON) {
            printf("%d-%f", n, z[n]);
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}