[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative 'Language/Expression'
require_relative 'Language/Intrinsics'
require_relative 'Language/Vectors'
require_relative 'Language/Slices'
require_relative 'Language/Dimension'
require_relative 'Language/Array'
require_relative 'Language/For'
require_relative 'Language/If'
require_relative 'Language/Case'
require_relative 'Language/While'
require_relative 'Language/Procedure'
require_relative 'Language/FuncCall'
require_relative 'Language/Annotation'
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
    set_lang(HIP)
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
      set_lang(HIP)
      assert_subprocess_output( "long long a;\n", "", &block )
      set_lang(CL)
      assert_subprocess_output( "long a;\n", "", &block )
    }
  end

  def test_decl_intptr_t
    block = lambda { decl Intptrt(:a) }
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4) :: a\n", "", &block )
    [C, CUDA, CL, HIP].each { |l|
      set_lang l
      assert_subprocess_output( "intptr_t a;\n", "", &block )
    }
    block = lambda { decl Intptrt(:a, signed: false) }
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4) :: a\n", "", &block )
    [C, CUDA, CL, HIP].each { |l|
      set_lang l
      assert_subprocess_output( "uintptr_t a;\n", "", &block )
    }
  end

  def test_decl_pointer
    block = lambda { decl Pointer(:a ) }
    [C, CUDA, CL, HIP].each { |l|
      set_lang l
      assert_subprocess_output( "void * a;\n", "", &block )
    }
    set_lang(FORTRAN)
    assert_raises("Pointers are unsupported in Fortran!", &block )
    push_env(:default_real_size => 8) {
      block = lambda { decl Pointer(:a, type: Real ) }
      [C, CUDA, CL, HIP].each { |l|
        set_lang l
        assert_subprocess_output( "double * a;\n", "", &block )
      }
      set_lang(FORTRAN)
      assert_raises("Pointers are unsupported in Fortran!", &block )
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
    set_lang(HIP)
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
      set_lang(HIP)
      assert_subprocess_output( "4.7f\n", "", &block )
      set_lang(CL)
      assert_subprocess_output( "4.7f\n", "", &block )
    }
  end

end
