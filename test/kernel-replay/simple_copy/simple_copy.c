void simple_copy(size_t n, const double *a, double *b) {
  for(size_t i = 0; i < n; i++) {
    b[i] = a[i];
  }
}
