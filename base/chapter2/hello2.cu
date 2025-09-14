#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("print from GTX1070 gpu\n");
}

int main(void) {
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
