require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestProcedure < Minitest::Test

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
    p = Procedure("minimum", [a,b], [], :return => c) { pr c === Ternary( a < b, a, b) }
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
