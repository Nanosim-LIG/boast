__global__ void simple_copy(const double *a, double *b) {
  size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  b[i] = a[i];
}
