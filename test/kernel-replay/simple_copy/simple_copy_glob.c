double a[1024];

void simple_copy(double *b) {
  for(size_t i = 0; i < 1024; i++) {
    b[i] = a[i];
  }
}
