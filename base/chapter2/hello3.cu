#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("print from GTX1070 gpu\n");
}

int main(void) {
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
