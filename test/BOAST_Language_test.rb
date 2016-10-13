[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative 'Language/Expression'
require_relative 'Language/Intrinsics'
require_relative 'Language/Vectors'
require_relative 'Language/Dimension'
require_relative 'Language/Array'
require_relative 'Language/For'
require_relative 'Language/If'
require_relative 'Language/Case'
require_relative 'Language/While'
require_relative 'Language/Procedure'
require_relative 'Language/FuncCall'
require_relative 'helper'

class TestLanguage < Minitest::Test

  def test_decl_int
    block = lambda { decl Int(:a) }
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4) :: a\n", "", &block )
    set_lang(C)
    assert_subprocess_output( "int32_t a;\n", "", &block )
    set_lang(CUDA)
    assert_subprocess_output( "int a;\n", "", &block )
    set_lang(CL)
    assert_subprocess_output( "int a;\n", "", &block )
  end

  def test_decl_int_64
    push_env(:default_int_size => 8) {
      block = lambda { decl Int(:a) }
      set_lang(FORTRAN)
      assert_subprocess_output( "integer(kind=8) :: a\n", "", &block )
      set_lang(C)
      assert_subprocess_output( "int64_t a;\n", "", &block )
      set_lang(CUDA)
      assert_subprocess_output( "long long a;\n", "", &block )
      set_lang(CL)
      assert_subprocess_output( "long a;\n", "", &block )
    }
  end

  def test_puts_float_64
    block = lambda { puts 4.7.to_var }
    set_lang(FORTRAN)
    assert_subprocess_output( "4.7_wp\n", "", &block )
    set_lang(C)
    assert_subprocess_output( "4.7\n", "", &block )
    set_lang(CUDA)
    assert_subprocess_output( "4.7\n", "", &block )
    set_lang(CL)
    assert_subprocess_output( "4.7\n", "", &block )
  end

  def test_puts_float_32
    push_env(:default_real_size => 4) {
      block = lambda { puts 4.7.to_var }
      set_lang(FORTRAN)
      assert_subprocess_output( "4.7\n", "", &block )
      set_lang(C)
      assert_subprocess_output( "4.7f\n", "", &block )
      set_lang(CUDA)
      assert_subprocess_output( "4.7f\n", "", &block )
      set_lang(CL)
      assert_subprocess_output( "4.7f\n", "", &block )
    }
  end

end
