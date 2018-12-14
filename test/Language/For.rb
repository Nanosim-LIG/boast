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

  def test_pr_block_for
    i = Int("i")
    n = Int("n")
    a = Int("a", :dim => Dim(n))
    f = For(i, 1, n)
    l = lambda { pr a[i] === i }
    block = lambda { pr f, &l }
    begin
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &block )
do i = 1, n, 1
  a(i) = i
end do
EOF
      [C, CL, CUDA].each { |lg|
        set_lang(lg)
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

  def test_pr_for_downward
    i = Int("i")
    a = Int("a", :dim => Dim())
    f = For(i, 5, 0, :step => -1) { pr a[i] === i }
    block = lambda { pr f }
    begin
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &block )
do i = 5, 0, -1
  a(i) = i
end do
EOF
      [C, CL, CUDA].each { |l|
        set_lang(l)
        assert_subprocess_output( <<EOF, "", &block )
for (i = 5; i >= 0; i += -1) {
  a[i - (1)] = i;
}
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_for_args
    i = Int("i")
    n = Int("n")
    a = Int("a", :dim => Dim(n))
    f = For(i, 1, n) { |x| pr a[i] === i * x }
    y = nil
    block1 = lambda { pr f, y }
    block2 = lambda { pr f[y] }
    begin
      [block1, block2].each { |block|
        y = rand(100)
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "", &block )
do i = 1, n, 1
  a(i) = (i) * (#{y})
end do
EOF
        [C, CL, CUDA].each { |l|
          y = rand(100)
          set_lang(l)
          assert_subprocess_output( <<EOF, "", &block )
for (i = 1; i <= n; i += 1) {
  a[i - (1)] = (i) * (#{y});
}
EOF
        }
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

  def test_for_unroll_param_true
    i = Int("i")
    a = Int("a", :dim => Dim(3))
    f = For(i, 1, 3) { pr a[i] === i }
    block1 = lambda { pr f.unroll(true) }
    block2 = lambda { pr f.unroll(true).unroll(true) }
    [block1, block2].each { |block|
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
    }
  end

  def test_for_unroll_param_false
    i = Int("i")
    a = Int("a", :dim => Dim(3))
    f = For(i, 1, 3) { pr a[i] === i }
    block1 = lambda { pr f.unroll(false) }
    block2 = lambda { pr f.unroll(true).unroll(false) }
    [block1, block2].each { |block|
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &block )
do i = 1, 3, 1
  a(i) = i
end do
EOF
      [C, CL, CUDA].each { |l|
        set_lang(l)
        assert_subprocess_output( <<EOF, "", &block )
for (i = 1; i <= 3; i += 1) {
  a[i - (1)] = i;
}
EOF
      }
    }
  end

  def test_for_unroll
    i = Int("i")
    a = Int("a", :dim => Dim(3))
    f = For(i, 1, 3) { pr a[i] === i }
    block = lambda { pr f.unroll }
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

  def test_for_unroll_integer
    i = Int("i")
    n = Int("n")
    a = Int("a", :dim => Dim(n))
    f = For(i, 1, n) { pr a[i] === i }
    block = lambda { pr f.unroll(2) }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
do i = 1, n - (1), 2
  a(i + 0) = i + 0
  a(i + 1) = i + 1
end do
do i = 1 + ((n + 0) / (2)) * (2), n, 1
  a(i) = i
end do
EOF
    [C, CL, CUDA].each { |l|
      set_lang(l)
      assert_subprocess_output( <<EOF, "", &block )
for (i = 1; i <= n - (1); i += 2) {
  a[i + 0 - (1)] = i + 0;
  a[i + 1 - (1)] = i + 1;
}
for (i = 1 + ((n + 0) / (2)) * (2); i <= n; i += 1) {
  a[i - (1)] = i;
}
EOF
    }
  end

  def test_for_unroll_block
    i = Int("i")
    a = Int("a", :dim => Dim(3))
    f = For(i, 1, 3)
    l = lambda { pr a[i] === i }
    block = lambda { pr f.unroll, &l }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
a(1) = 1
a(2) = 2
a(3) = 3
EOF
    [C, CL, CUDA].each { |lg|
      set_lang(lg)
      assert_subprocess_output( <<EOF, "", &block )
a[1 - (1)] = 1;
a[2 - (1)] = 2;
a[3 - (1)] = 3;
EOF
    }
  end

  def test_for_unroll_integer_block
    i = Int("i")
    n = Int("n")
    a = Int("a", :dim => Dim(n))
    f = For(i, 1, n)
    l = lambda { pr a[i] === i }
    block = lambda { pr f.unroll(2), &l }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
do i = 1, n - (1), 2
  a(i + 0) = i + 0
  a(i + 1) = i + 1
end do
do i = 1 + ((n + 0) / (2)) * (2), n, 1
  a(i) = i
end do
EOF
    [C, CL, CUDA].each { |lg|
      set_lang(lg)
      assert_subprocess_output( <<EOF, "", &block )
for (i = 1; i <= n - (1); i += 2) {
  a[i + 0 - (1)] = i + 0;
  a[i + 1 - (1)] = i + 1;
}
for (i = 1 + ((n + 0) / (2)) * (2); i <= n; i += 1) {
  a[i - (1)] = i;
}
EOF
    }
  end

  def test_for_unroll_args
    i = Int("i")
    a = Int("a", :dim => Dim(3))
    f = For(i, 1, 3) { |x| pr a[i] === i * x }
    block = lambda { pr f.unroll[2] }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
a(1) = (1) * (2)
a(2) = (2) * (2)
a(3) = (3) * (2)
EOF
    [C, CL, CUDA].each { |l|
      set_lang(l)
      assert_subprocess_output( <<EOF, "", &block )
a[1 - (1)] = (1) * (2);
a[2 - (1)] = (2) * (2);
a[3 - (1)] = (3) * (2);
EOF
    }
  end

  def test_for_unroll_integer_args
    i = Int("i")
    n = Int("n")
    a = Int("a", :dim => Dim(n))
    f = For(i, 1, n) { |x| pr a[i] === i * x }
    block = lambda { pr f.unroll(2)[3] }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
do i = 1, n - (1), 2
  a(i + 0) = (i + 0) * (3)
  a(i + 1) = (i + 1) * (3)
end do
do i = 1 + ((n + 0) / (2)) * (2), n, 1
  a(i) = (i) * (3)
end do
EOF
    [C, CL, CUDA].each { |l|
      set_lang(l)
      assert_subprocess_output( <<EOF, "", &block )
for (i = 1; i <= n - (1); i += 2) {
  a[i + 0 - (1)] = (i + 0) * (3);
  a[i + 1 - (1)] = (i + 1) * (3);
}
for (i = 1 + ((n + 0) / (2)) * (2); i <= n; i += 1) {
  a[i - (1)] = (i) * (3);
}
EOF
    }
  end

end
