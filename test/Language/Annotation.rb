require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestAnnotation < Minitest::Test

  def test_annotate_expression
    push_env( :lang => C, :annotate => true, :annotate_list => [ "Expression" ], :annotate_indepth_list => [ ], :annotate_level => 2 ) {
      i = Int("i")
      n = Int("n")
      expr = n + i * 2
      block = lambda { pr expr }
      assert_subprocess_output( <<EOF, "", &block )
/* --- */
/* Expression0:  */
n + (i) * (2);
EOF
      push_env( :annotate_indepth_list => [ "Expression" ], :annotate_level => 0 ) {
        assert_subprocess_output( <<EOF, "", &block )
/* --- */
/* Expression1: */
/*   :operator: BOAST::Addition */
/*   :operand1: n */
/*   :operand2: \"(i) * (2)\" */
n + (i) * (2);
EOF
        push_env( :annotate_level => 1 ) {
          assert_subprocess_output( <<EOF, "", &block )
/* --- */
/* Expression3: */
/*   :operator: BOAST::Addition */
/*   :operand1: n */
/*   :operand2: */
/*     Expression2: */
/*       :operator: BOAST::Multiplication */
/*       :operand1: i */
/*       :operand2: 2 */
n + (i) * (2);
EOF
        }
      }
    }
    reset_annotate_numbers
  end

  def test_reset_annotate_numebrs
    push_env( :lang => C, :annotate => true, :annotate_list => [ "For" ], :annotate_indepth_list => [ "For", "Expression" ], :annotate_level => 2 ) {
      i = Int("i")
      n = Int("n")
      a = Int("a", :dim => Dim(0,n-1))
      f = For(i, 0, n - 1) { pr a[i] === i }
      block = lambda { pr f }
      2.times {
        assert_subprocess_output( <<EOF, "", &block )
/* --- */
/* For0: */
/*   :iterator: i */
/*   :first: 0 */
/*   :last: */
/*     Expression0: */
/*       :operator: BOAST::Subtraction */
/*       :operand1: n */
/*       :operand2: 1 */
/*   :step: 1 */
/*   :operator: \"<=\" */
for (i = 0; i <= n - (1); i += 1) {
  a[i] = i;
}
EOF
        reset_annotate_numbers
      }
    }
  end

  def test_for_annotate
    push_env( :lang => C, :annotate => true, :annotate_list => [ "For" ], :annotate_indepth_list => [ "For" ], :annotate_level => 1 ) {
      i = Int("i")
      n = Int("n")
      a = Int("a", :dim => Dim(0,n-1))
      f = For(i, 0, n - 1) { pr a[i] === i }
      block = lambda { pr f }
      assert_subprocess_output( <<EOF, "", &block )
/* --- */
/* For0: */
/*   :iterator: i */
/*   :first: 0 */
/*   :last: n - (1) */
/*   :step: 1 */
/*   :operator: \"<=\" */
for (i = 0; i <= n - (1); i += 1) {
  a[i] = i;
}
EOF
      push_env( :lang => FORTRAN ) {
        assert_subprocess_output( <<EOF, "", &block )
! ---
! For1:
!   :iterator: i
!   :first: 0
!   :last: n - (1)
!   :step: 1
!   :operator: \"<=\"
do i = 0, n - (1), 1
  a(i) = i
end do
EOF
      }
      push_env( :annotate_list => [ "For" ], :annotate_indepth_list => [ "For", "Expression" ], :annotate_level => 2 ) {
        assert_subprocess_output( <<EOF, "", &block )
/* --- */
/* For2: */
/*   :iterator: i */
/*   :first: 0 */
/*   :last: */
/*     Expression0: */
/*       :operator: BOAST::Subtraction */
/*       :operand1: n */
/*       :operand2: 1 */
/*   :step: 1 */
/*   :operator: \"<=\" */
for (i = 0; i <= n - (1); i += 1) {
  a[i] = i;
}
EOF
        push_env( :lang => FORTRAN ) {
          assert_subprocess_output( <<EOF, "", &block )
! ---
! For3:
!   :iterator: i
!   :first: 0
!   :last:
!     Expression1:
!       :operator: BOAST::Subtraction
!       :operand1: n
!       :operand2: 1
!   :step: 1
!   :operator: \"<=\"
do i = 0, n - (1), 1
  a(i) = i
end do
EOF
        }
      }
      reset_annotate_numbers
    }
  end

end
