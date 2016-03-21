require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestFuncCall < Minitest::Test

  def test_pr_funccall
    a = Int :a
    b = Int :b
    c = Int :c
    block = lambda { pr c === FuncCall(:min, a, b) }
    begin
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &block )
c = min(a, b)
EOF
      [C, CL, CUDA].each { |l|
        set_lang(l)
        assert_subprocess_output( <<EOF, "", &block )
c = min(a, b);
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_typed_funccall
    a = Int :a, :vector_length => 4
    b = Int :b, :vector_length => 4
    c = Real :c, :size => 4, :vector_length => 4
    block = lambda { pr c === FuncCall(:min, a, b, :returns => a ) }
    begin
      set_lang(C)
      push_env(:architecture => X86, :model => :nehalem) {
        assert_subprocess_output( <<EOF, "", &block )
c = _mm_cvtepi32_ps( min(a, b) );
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_registered_funccall
    a = Int :a
    b = Int :b
    c = Int :c
    register_funccall(:min)
    block = lambda { pr c === min(a, b) }
    begin
      set_lang(FORTRAN)
      assert_subprocess_output( <<EOF, "", &block )
c = min(a, b)
EOF
      [C, CL, CUDA].each { |l|
        set_lang(l)
        assert_subprocess_output( <<EOF, "", &block )
c = min(a, b);
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

  def test_pr_typed_registered_funccall
    a = Int :a, :vector_length => 4
    b = Int :b, :vector_length => 4
    c = Real :c, :size => 4, :vector_length => 4
    register_funccall("min", :returns => a)
    block = lambda { pr c === min(a, b) }
    begin
      set_lang(C)
      push_env(:architecture => X86, :model => :nehalem) {
        assert_subprocess_output( <<EOF, "", &block )
c = _mm_cvtepi32_ps( min(a, b) );
EOF
      }
    ensure
      set_indent_level(0)
    end
  end

end
