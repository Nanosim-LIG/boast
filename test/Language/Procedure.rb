require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestProcedure < Minitest::Test

  def test_procedure_local_variables_aligned
    n = Int( :n, :dir => :in  )
    a = Int( :a, :dir => :in, :dim => Dim(n), :align => 32 )
    b = Int( :b, :dir => :in, :dim => Dim(n), :align => 32  )
    c = Int( :c, :dir => :out, :dim => Dim(n), :align => 32  )
    i = Int( :i )
    p = Procedure("vector_add", [n,a,b,c], :locals => [i]) { pr For(i,1,n) { pr c[i] === a[i] + b[i] } }
    block = lambda { pr p }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
SUBROUTINE vector_add(n, a, b, c)
  integer, parameter :: wp=kind(1.0d0)
  integer(kind=4), intent(in) :: n
  integer(kind=4), intent(in), dimension(n) :: a
  integer(kind=4), intent(in), dimension(n) :: b
  integer(kind=4), intent(out), dimension(n) :: c
  integer(kind=4) :: i
!DIR$ ASSUME_ALIGNED a: 32
!DIR$ ASSUME_ALIGNED b: 32
!DIR$ ASSUME_ALIGNED c: 32
  do i = 1, n, 1
    c(i) = a(i) + b(i)
  end do
END SUBROUTINE vector_add
EOF
    push_env( :lang => C, :model => :ivybridge ) {
      assert_subprocess_output( <<EOF, "", &block )
void vector_add(const int32_t n, const int32_t * a, const int32_t * b, int32_t * c){
  __assume_aligned(a, 32);
  __assume_aligned(b, 32);
  __assume_aligned(c, 32);
  int32_t i;
  for (i = 1; i <= n; i += 1) {
    c[i - (1)] = a[i - (1)] + b[i - (1)];
  }
}
EOF
    }
  end

  def test_procedure_local_variables_vectors
    a = Int( :a, :dir => :in, :vector_length => 4 )
    b = Int( :b, :dir => :in, :vector_length => 4  )
    c = Int( :c, :dir => :out, :vector_length => 4  )
    i = Int( :i, :vector_length => 4  )
    p = Procedure("fma", [a,b,c], :locals => [i]) { pr i === FMA(a,b,c); pr c === i }
    block = lambda { pr p }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
SUBROUTINE fma(a, b, c)
  integer, parameter :: wp=kind(1.0d0)
  integer(kind=4), intent(in), dimension(4) :: a
  integer(kind=4), intent(in), dimension(4) :: b
  integer(kind=4), intent(out), dimension(4) :: c
  integer(kind=4), dimension(4) :: i
  !DIR$ ATTRIBUTES ALIGN: 16:: i
!DIR$ ASSUME_ALIGNED a: 16
!DIR$ ASSUME_ALIGNED b: 16
!DIR$ ASSUME_ALIGNED c: 16
  i = c + (a) * (b)
  c = i
END SUBROUTINE fma
EOF
    push_env( :lang => C, :model => :ivybridge ) {
      assert_subprocess_output( <<EOF, "", &block )
void fma(const __m128i a, const __m128i b, __m128i * c){
  __m128i i;
  i = _mm_add_epi32( (*c), _mm_mullo_epi32( a, b ) );
  (*c) = i;
}
EOF
    }
  end

  def test_procedure_local_variables
    a = Int( :a, :dir => :in )
    b = Int( :b, :dir => :in )
    c = Int( :c, :dir => :out )
    i = Int( :i )
    p = Procedure("minimum", [a,b,c], :locals => [i]) { pr i === Ternary( a < b, a, b); pr c === i }
    block = lambda { pr p }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
SUBROUTINE minimum(a, b, c)
  integer, parameter :: wp=kind(1.0d0)
  integer(kind=4), intent(in) :: a
  integer(kind=4), intent(in) :: b
  integer(kind=4), intent(out) :: c
  integer(kind=4) :: i
  i = merge(a, b, a < b)
  c = i
END SUBROUTINE minimum
EOF
    set_lang(C)
    assert_subprocess_output( <<EOF, "", &block )
void minimum(const int32_t a, const int32_t b, int32_t * c){
  int32_t i;
  i = (a < b ? a : b);
  (*c) = i;
}
EOF
  end

  def test_procedure
    a = Int( :a, :dir => :in )
    b = Int( :b, :dir => :in )
    c = Int( :c, :dir => :out )
    p = Procedure("minimum", [a,b,c]) { pr c === Ternary( a < b, a, b) }
    block = lambda { pr p }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
SUBROUTINE minimum(a, b, c)
  integer, parameter :: wp=kind(1.0d0)
  integer(kind=4), intent(in) :: a
  integer(kind=4), intent(in) :: b
  integer(kind=4), intent(out) :: c
  c = merge(a, b, a < b)
END SUBROUTINE minimum
EOF
    set_lang(C)
    assert_subprocess_output( <<EOF, "", &block )
void minimum(const int32_t a, const int32_t b, int32_t * c){
  (*c) = (a < b ? a : b);
}
EOF
  end

  def test_function
    a = Int( :a, :dir => :in )
    b = Int( :b, :dir => :in )
    c = Int( :c )
    p = Procedure("minimum", [a,b], :return => c) { pr c === Ternary( a < b, a, b) }
    block = lambda { pr p }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
integer(kind=4) FUNCTION minimum(a, b)
  integer, parameter :: wp=kind(1.0d0)
  integer(kind=4), intent(in) :: a
  integer(kind=4), intent(in) :: b
  integer(kind=4) :: c
  c = merge(a, b, a < b)
  minimum = c
END FUNCTION minimum
EOF
    set_lang(C)
    assert_subprocess_output( <<EOF, "", &block )
int32_t minimum(const int32_t a, const int32_t b){
  int32_t c;
  c = (a < b ? a : b);
  return c;
}
EOF
  end

end
