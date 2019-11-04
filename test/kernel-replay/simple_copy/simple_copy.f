SUBROUTINE simple_copy(n, a, b)
  integer(kind=8) :: n
  real(kind=8), dimension(*), intent(in) :: a
  real(kind=8), dimension(*), intent(out) :: b
  integer(kind=8) :: i
  
  do i = 1, n
    b(i) = a(i)
  end do
END SUBROUTINE simple_copy

