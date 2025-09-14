#include <stdio.h>

__global__ void hello_from_gpu() {
    const int bid = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("block %d and thread (%d %d)--> print from GTX1070 gpu\n",bid, tx, ty);
}

int main(void) {
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
