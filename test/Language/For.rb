require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestFor < Minitest::Test

  def test_pr_for
    i = Int("i")
    n = Int("n")
    a = Int("a", :dim => Dim(n))
    f = For(i, 1, n) { pr a[i] === i }
    block = lambda { pr f }
    begin
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &block )
do i = 1, n, 1
  a(i) = i
end do
EOF
      [C, CL, CUDA].each { |l|
        set_lang(l)
        assert_subprocess_output( <<EOF, "", &block )
for (i = 1; i <= n; i += 1) {
  a[i - (1)] = i;
}
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_opn_close_for
    i = Int("i")
    n = Int("n")
    a = Int("a", :dim => Dim(n))
    f = For(i, 1, n)
    block1 = lambda { opn f }
    block2 = lambda { pr a[i] === i }
    block3 = lambda { close f }
    begin
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &block1 )
do i = 1, n, 1
EOF
      assert_subprocess_output( <<EOF, "", &block2 )
  a(i) = i
EOF
      assert_subprocess_output( <<EOF, "", &block3 )
end do
EOF
      [C, CL, CUDA].each { |l|
        set_lang(l)
        assert_subprocess_output( <<EOF, "", &block1 )
for (i = 1; i <= n; i += 1) {
EOF
        assert_subprocess_output( <<EOF, "", &block2 )
  a[i - (1)] = i;
EOF
        assert_subprocess_output( <<EOF, "", &block3 )
}
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_for_unroll
    i = Int("i")
    a = Int("a", :dim => Dim(3))
    f = For(i, 1, 3) { pr a[i] === i }
    block = lambda { f.unroll }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
a(1) = 1
a(2) = 2
a(3) = 3
EOF
    [C, CL, CUDA].each { |l|
      set_lang(l)
      assert_subprocess_output( <<EOF, "", &block )
a[1 - (1)] = 1;
a[2 - (1)] = 2;
a[3 - (1)] = 3;
EOF
    }
  end

end
