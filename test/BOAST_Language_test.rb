[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'minitest/autorun'
require 'BOAST'
include BOAST

module MiniTest::Assertions

  def assert_subprocess_output( stdout = nil, stderr = nil)
    out, err = capture_subprocess_io do
      yield
    end

    err_msg = Regexp === stderr ? :assert_match : :assert_equal if stderr
    out_msg = Regexp === stdout ? :assert_match : :assert_equal if stdout

    y = send err_msg, stderr, err, "In stderr" if err_msg
    x = send out_msg, stdout, out, "In stdout" if out_msg

    (!stdout || x) && (!stderr || y)
  end

end

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

  def test_decl_int_array
    n1 = Int("n1")
    arr = Int(:a, :dim => [Dim(5,15), Dim(n1)])
    block = lambda { decl arr }
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4), dimension(5:15, n1) :: a\n", "", &block )
    set_lang(C)
    assert_subprocess_output( "int32_t * a;\n", "", &block )
    begin
      push_env(:use_vla => true)
      assert_subprocess_output( "int32_t a[n1][11];\n", "", &block )
    ensure
      pop_env(:use_vla)
    end
  end

  def test_decl_int_array_unkwown_dim
    arr = Int(:a, :dim => [Dim(5,15), Dim()])
    block = lambda { decl arr }
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4), dimension(5:15, *) :: a\n", "", &block )
    set_lang(C)
    assert_subprocess_output( "int32_t * a;\n", "", &block )
    begin
      push_env(:use_vla => true)
      assert_subprocess_output( "int32_t a[][11];\n", "", &block )
    ensure
      pop_env(:use_vla)
    end
  end

  def test_decl_int_array_deffered_shape
    arr = Int(:a, :dim => [Dim(5,15), Dim()], :deferred_shape => true)
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4), dimension(:, :) :: a\n", "" ) do
      decl arr
    end
  end

  def test_pr_int_array_index
    n1 = Int("n1")
    arr = Int(:a, :dim => [Dim(5,15), Dim(n1)])
    block = lambda { pr arr[6,7] }
    set_lang(FORTRAN)
    assert_subprocess_output( "a(6, 7)\n", "", &block )
    set_lang(C)
    assert_subprocess_output( "a[6 - (5) + (11) * (7 - (1))];\n", "", &block )
    begin
      push_env(:use_vla => true)
      assert_subprocess_output( "a[7 - (1)][6 - (5)];\n", "", &block )
    ensure
      pop_env(:use_vla)
    end
  end

  def test_pr_int_array_index_unkwown_dim
    n1 = Int("n1")
    arr = Int(:a, :dim => [Dim(n1), Dim()])
    block = lambda { pr arr[6,7] }
    set_lang(FORTRAN)
    assert_subprocess_output( "a(6, 7)\n", "", &block )
    set_lang(C)
    assert_subprocess_output( "a[6 - (1) + (n1) * (7 - (1))];\n", "", &block )
    begin
      push_env(:use_vla => true, :array_start => 0)
      assert_subprocess_output( "a[7][6];\n", "", &block )
    ensure
      pop_env(:use_vla, :array_start)
    end
  end

end
