require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestIf < Minitest::Test

  def setup
    i = Int("i")
    n = Int("n")
    a = Int("a", :dim => Dim(n))
    @cond1  = i <  n
    @cond2  = i == n
    @block1 = lambda { pr a[i] ===  i }
    @block2 = lambda { pr a[i] ===  0 }
    @block3 = lambda { pr a[i] === -i }
    @block4 = lambda { |x| pr a[i] ===  i * x }
    @block5 = lambda { |x| pr a[i] ===  0 * x }
    @block6 = lambda { |x| pr a[i] === -i * x }
  end

  def test_pr_if
    f1 = If(@cond1 =>  @block1)
    f2 = If(@cond1, &@block1)
    begin
      [f1, f2].each { |f|
        block = lambda { pr f }
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "", &block )
if (i < n) then
  a(i) = i
end if
EOF
        [C, CL, CUDA, HIP].each { |l|
          set_lang(l)
          assert_subprocess_output( <<EOF, "", &block )
if (i < n) {
  a[i - (1)] = i;
}
EOF
        }
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_if_else
    f1 = If(@cond1 => @block1, :else => @block3)
    f2 = If(@cond1 => @block1, &@block3)
    begin
      [f1, f2].each { |f|
        block = lambda { pr f }
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "", &block )
if (i < n) then
  a(i) = i
else
  a(i) =  -(i)
end if
EOF
        [C, CL, CUDA, HIP].each { |l|
          set_lang(l)
          assert_subprocess_output( <<EOF, "", &block )
if (i < n) {
  a[i - (1)] = i;
} else {
  a[i - (1)] =  -(i);
}
EOF
        }
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_if_elsif_else
    f1 = If(@cond1 => @block1, @cond2 => @block2, :else => @block3)
    f2 = If(@cond1 => @block1, @cond2 => @block2, &@block3)
    begin
      [f1, f2].each { |f|
        block = lambda { pr f }
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "", &block )
if (i < n) then
  a(i) = i
else if (i == n) then
  a(i) = 0
else
  a(i) =  -(i)
end if
EOF
        [C, CL, CUDA, HIP].each { |l|
          set_lang(l)
          assert_subprocess_output( <<EOF, "", &block )
if (i < n) {
  a[i - (1)] = i;
} else if (i == n) {
  a[i - (1)] = 0;
} else {
  a[i - (1)] =  -(i);
}
EOF
        }
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_if_elsif_else_args
    f1 = If(@cond1 => @block4, @cond2 => @block5, :else => @block6)
    f2 = If(@cond1 => @block4, @cond2 => @block5, &@block6)
    y = nil
    begin
      [f1, f2].each { |f|
        y = rand(100)
        block = lambda { pr f[y] }
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "", &block )
if (i < n) then
  a(i) = (i) * (#{y})
else if (i == n) then
  a(i) = 0
else
  a(i) = ( -(i)) * (#{y})
end if
EOF
        [C, CL, CUDA, HIP].each { |l|
          y = rand(100)
          set_lang(l)
          assert_subprocess_output( <<EOF, "", &block )
if (i < n) {
  a[i - (1)] = (i) * (#{y});
} else if (i == n) {
  a[i - (1)] = 0;
} else {
  a[i - (1)] = ( -(i)) * (#{y});
}
EOF
        }
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_opn_close_if
    f = If(@cond1)
    opn_block   = lambda { opn   f }
    close_block = lambda { close f }
    begin
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &opn_block )
if (i < n) then
EOF
      assert_subprocess_output( <<EOF, "", &@block1 )
  a(i) = i
EOF
      assert_subprocess_output( <<EOF, "", &close_block )
end if
EOF
      [C, CL, CUDA, HIP].each { |l|
        set_lang(l)
        assert_subprocess_output( <<EOF, "", &opn_block )
if (i < n) {
EOF
      assert_subprocess_output( <<EOF, "", &@block1 )
  a[i - (1)] = i;
EOF
      assert_subprocess_output( <<EOF, "", &close_block )
}
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_open_open_close_if
    f = If(@cond1 => nil, @cond2 => nil, :else => nil)
    opn_block = lambda { opn f }
    open1_block = lambda { f.open(1) }
    open2_block = lambda { f.open(2) }
    close_block = lambda { close f }
    begin
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &opn_block )
if (i < n) then
EOF
      assert_subprocess_output( <<EOF, "", &@block1 )
  a(i) = i
EOF
      assert_subprocess_output( <<EOF, "", &open1_block )
else if (i == n) then
EOF
      assert_subprocess_output( <<EOF, "", &@block2 )
  a(i) = 0
EOF
      assert_subprocess_output( <<EOF, "", &open2_block )
else
EOF
      assert_subprocess_output( <<EOF, "", &@block3 )
  a(i) =  -(i)
EOF
      assert_subprocess_output( <<EOF, "", &close_block )
end if
EOF
      [C, CL, CUDA, HIP].each { |l|
        set_lang(l)
        assert_subprocess_output( <<EOF, "", &opn_block )
if (i < n) {
EOF
        assert_subprocess_output( <<EOF, "", &@block1 )
  a[i - (1)] = i;
EOF
        assert_subprocess_output( <<EOF, "", &open1_block )
} else if (i == n) {
EOF
        assert_subprocess_output( <<EOF, "", &@block2 )
  a[i - (1)] = 0;
EOF
        assert_subprocess_output( <<EOF, "", &open2_block )
} else {
EOF
        assert_subprocess_output( <<EOF, "", &@block3 )
  a[i - (1)] =  -(i);
EOF
        assert_subprocess_output( <<EOF, "", &close_block )
}
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

end
