[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative 'Language/Array'
require_relative 'helper'

class TestLanguage < Minitest::Test

  def test_decl_int
    block = lambda { decl Int(:a) }
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4) :: a\n", "", &block )
    set_lang(C)
    assert_subprocess_output( "int32_t a;\n", "", &block )
  end

  def test_decl_int_64
    begin
      push_env(:default_int_size => 8)
      block = lambda { decl Int(:a) }
      set_lang(FORTRAN)
      assert_subprocess_output( "integer(kind=8) :: a\n", "", &block )
      set_lang(C)
      assert_subprocess_output( "int64_t a;\n", "", &block )
    ensure
      pop_env(:default_int_size)
    end
  end

end
