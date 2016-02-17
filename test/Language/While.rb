require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestWhile < Minitest::Test

  def test_pr_while
    i = Int("i")
    n = Int("n")
    a = Int("a", :dim => Dim(n))
    w = While(i < n) { pr a[i] === i }
    block = lambda { pr w }
    begin
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &block )
do while (i < n)
  a(i) = i
end do
EOF
      [C, CL, CUDA].each { |l|
        set_lang(l)
        assert_subprocess_output( <<EOF, "", &block )
while (i < n) {
  a[i - (1)] = i;
}
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_while_args
    i = Int("i")
    n = Int("n")
    a = Int("a", :dim => Dim(n))
    w = While(i < n) { |x| pr a[i] === i * x }
    y = nil
    block1 = lambda { pr w, y }
    block2 = lambda { pr w[y] }
    begin
      [block1, block2].each { |block|
        y = rand(100)
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "", &block )
do while (i < n)
  a(i) = (i) * (#{y})
end do
EOF
        [C, CL, CUDA].each { |l|
          y = rand(100)
          set_lang(l)
          assert_subprocess_output( <<EOF, "", &block )
while (i < n) {
  a[i - (1)] = (i) * (#{y});
}
EOF
        }
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_opn_close_while
    i = Int("i")
    n = Int("n")
    a = Int("a", :dim => Dim(n))
    w = While(i < n)
    block1 = lambda { opn w }
    block2 = lambda { pr a[i] === i }
    block3 = lambda { close w }
    begin
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &block1 )
do while (i < n)
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
while (i < n) {
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

end
