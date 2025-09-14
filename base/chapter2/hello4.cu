#include <stdio.h>

__global__ void hello_from_gpu() {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("block %d and thread %d --> print from GTX1070 gpu\n",bid, tid);
}

int main(void) {
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
