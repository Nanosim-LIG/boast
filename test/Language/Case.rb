require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestCase < Minitest::Test

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
    c = Case(@expr, @const1 =>  @block1)
    begin
      block = lambda { pr c }
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &block )
select case ((n) * (2))
  case (2)
    a(i) = i
end select
EOF
      [C, CL, CUDA].each { |l|
        set_lang(l)
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
    c1 = Case(@expr, @const1 => @block1, :default => @block2)
    c2 = Case(@expr, @const1 => @block1, &@block2)
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
        [C, CL, CUDA].each { |l|
          set_lang(l)
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
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_case_multiple_values_default
    c1 = Case(@expr, [@const1, @const2] => @block1, :default => @block2)
    c2 = Case(@expr, [@const1, @const2] => @block1, &@block2)
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
        [C, CL, CUDA].each { |l|
          set_lang(l)
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
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_multiple_case_default
    c1 = Case(@expr, @const1 => @block1, @const2 => @block3, :default => @block2)
    c2 = Case(@expr, @const1 => @block1, @const2 => @block3, &@block2)
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
        [C, CL, CUDA].each { |l|
          set_lang(l)
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
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_open_close_multiple_case_default
    c1 = Case(@expr, @const1 => @block1, @const2 => @block3, :default => @block2)
    c2 = Case(@expr, @const1 => @block1, @const2 => @block3, &@block2)
    begin
      [c1, c2].each { |c|
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "" ) { opn c }
select case ((n) * (2))
EOF
        assert_subprocess_output( <<EOF, "" ) { opn c.case_conditions[0] }
  case (2)
EOF
        assert_subprocess_output( <<EOF, "" ) { c.case_conditions[0].block.call }
    a(i) = i
EOF
        assert_subprocess_output( "", "" ) { close c.case_conditions[0] }
        assert_subprocess_output( <<EOF, "" ) { opn c.case_conditions[1] }
  case (8)
EOF
        assert_subprocess_output( <<EOF, "" ) { c.case_conditions[1].block.call }
    a(i) =  -(i)
EOF
        assert_subprocess_output( "", "" ) { close c.case_conditions[1] }
        assert_subprocess_output( <<EOF, "" ) { opn c.case_conditions[2] }
  case default
EOF
        assert_subprocess_output( <<EOF, "" ) { c.case_conditions[2].block.call }
    a(i) = 0
EOF
        assert_subprocess_output( "", "" ) { close c.case_conditions[2] }
        assert_subprocess_output( <<EOF, "" ) { close c }
end select
EOF
        [C, CL, CUDA].each { |l|
          set_lang(l)
          assert_subprocess_output( <<EOF, "" ) { opn c }
switch ((n) * (2)) {
EOF
          assert_subprocess_output( <<EOF, "" ) { opn c.case_conditions[0] }
  case 2 :
EOF
          assert_subprocess_output( <<EOF, "" ) { c.case_conditions[0].block.call }
    a[i - (1)] = i;
EOF
          assert_subprocess_output( <<EOF, "" ) { close c.case_conditions[0] }
    break;
EOF
          assert_subprocess_output( <<EOF, "" ) { opn c.case_conditions[1] }
  case 8 :
EOF
          assert_subprocess_output( <<EOF, "" ) { c.case_conditions[1].block.call }
    a[i - (1)] =  -(i);
EOF
          assert_subprocess_output( <<EOF, "" ) { close c.case_conditions[1] }
    break;
EOF
          assert_subprocess_output( <<EOF, "" ) { opn c.case_conditions[2] }
  default :
EOF
          assert_subprocess_output( <<EOF, "" ) { c.case_conditions[2].block.call }
    a[i - (1)] = 0;
EOF
          assert_subprocess_output( "", "" ) { close c.case_conditions[2] }
          assert_subprocess_output( <<EOF, "" ) { close c }
}
EOF
        }
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_multiple_case_default_by_line
    c1 = Case(@expr, @const1 => @block1, @const2 => @block3, :default => @block2)
    c2 = Case(@expr, @const1 => @block1, @const2 => @block3, &@block2)
    begin
      [c1, c2].each { |c|
        set_lang(FORTRAN)
        assert_subprocess_output( <<EOF, "" ) { opn c }
select case ((n) * (2))
EOF
        assert_subprocess_output( <<EOF, "" ) { pr c.case_conditions[0], &@block1 }
  case (2)
    a(i) = i
EOF
        assert_subprocess_output( <<EOF, "" ) { pr c.case_conditions[1], &@block3 }
  case (8)
    a(i) =  -(i)
EOF
        assert_subprocess_output( <<EOF, "" ) { pr c.case_conditions[2], &@block2 }
  case default
    a(i) = 0
EOF
        assert_subprocess_output( <<EOF, "" ) { close c }
end select
EOF
        [C, CL, CUDA].each { |l|
          set_lang(l)
          assert_subprocess_output( <<EOF, "" ) { opn c }
switch ((n) * (2)) {
EOF
          assert_subprocess_output( <<EOF, "" ) { pr c.case_conditions[0], &@block1 }
  case 2 :
    a[i - (1)] = i;
    break;
EOF
          assert_subprocess_output( <<EOF, "" ) { pr c.case_conditions[1], &@block3 }
  case 8 :
    a[i - (1)] =  -(i);
    break;
EOF
          assert_subprocess_output( <<EOF, "" ) { pr c.case_conditions[2], &@block2 }
  default :
    a[i - (1)] = 0;
EOF
          assert_subprocess_output( <<EOF, "" ) { close c }
}
EOF
        }
      }
    ensure
      set_indent_level(0)
    end
  end

end
