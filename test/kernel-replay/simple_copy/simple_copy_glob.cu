__device__ double a[1024];

__global__ void simple_copy(double *b) {
  size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  b[i] = a[i];
}
