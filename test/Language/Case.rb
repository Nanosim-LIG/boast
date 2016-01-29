require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestIf < Minitest::Test

  def setup
    n = Int("n")
    i = Int("i")
    a = Int("a", :dim => Dim(n))
    @expr  = n * 2
    @const1  = 2
    @const2  = 8
    @block1 = lambda { pr a[i] ===  i }
    @block2 = lambda { pr a[i] ===  0 }
    @block3 = lambda { pr a[i] === -i }
  end

  def test_pr_case
    c1 = Case(@expr, @const1,  @block1)
    c2 = Case(@expr, @const1, &@block1)
    begin
      [c1, c2].each { |c|
        block = lambda { pr c }
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "", &block )
select case ((n) * (2))
  case (2)
    a(i) = i
end select
EOF
        set_lang(C)
        assert_subprocess_output( <<EOF, "", &block )
switch ((n) * (2)) {
  case 2 :
    a[i - (1)] = i;
    break;
}
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_case_default
    c1 = Case(@expr, @const1, @block1,  @block2)
    c2 = Case(@expr, @const1, @block1, &@block2)
    begin
      [c1, c2].each { |c|
        block = lambda { pr c }
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "", &block )
select case ((n) * (2))
  case (2)
    a(i) = i
  case default
    a(i) = 0
end select
EOF
        set_lang(C)
        assert_subprocess_output( <<EOF, "", &block )
switch ((n) * (2)) {
  case 2 :
    a[i - (1)] = i;
    break;
  default :
    a[i - (1)] = 0;
}
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_case_default
    c1 = Case(@expr, [@const1, @const2], @block1,  @block2)
    c2 = Case(@expr, [@const1, @const2], @block1, &@block2)
    begin
      [c1, c2].each { |c|
        block = lambda { pr c }
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "", &block )
select case ((n) * (2))
  case (2, 8)
    a(i) = i
  case default
    a(i) = 0
end select
EOF
        set_lang(C)
        assert_subprocess_output( <<EOF, "", &block )
switch ((n) * (2)) {
  case 2 : case 8 :
    a[i - (1)] = i;
    break;
  default :
    a[i - (1)] = 0;
}
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_multiple_case_default
    c1 = Case(@expr, @const1, @block1, @const2, @block3,  @block2)
    c2 = Case(@expr, @const1, @block1, @const2, @block3, &@block2)
    begin
      [c1, c2].each { |c|
        block = lambda { pr c }
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "", &block )
select case ((n) * (2))
  case (2)
    a(i) = i
  case (8)
    a(i) =  -(i)
  case default
    a(i) = 0
end select
EOF
        set_lang(C)
        assert_subprocess_output( <<EOF, "", &block )
switch ((n) * (2)) {
  case 2 :
    a[i - (1)] = i;
    break;
  case 8 :
    a[i - (1)] =  -(i);
    break;
  default :
    a[i - (1)] = 0;
}
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

end
