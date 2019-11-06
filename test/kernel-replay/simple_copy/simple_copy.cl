kernel void simple_copy(global const double *a, global double *b) {
  size_t i = get_global_id(0);
  b[i] = a[i];
}
