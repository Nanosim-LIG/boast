[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'minitest/autorun'
require 'BOAST'
include BOAST

class TestLanguage < Minitest::Test

  def test_decl_int
    set_lang(FORTRAN)
    out, err = capture_subprocess_io do
      decl Int(:a)
    end
    assert_equal(out, "integer(kind=4) :: a\n")
    assert_equal(err, "")
    set_lang(C)
    out, err = capture_subprocess_io do
      decl Int(:a)
    end
    assert_equal(out, "int32_t a;\n")
    assert_equal(err, "")
  end

  def test_decl_int_64
    set_lang(FORTRAN)
    begin
      push_env(:default_int_size => 8)
      out, err = capture_subprocess_io do
        decl Int(:a)
      end
      assert_equal(out, "integer(kind=8) :: a\n")
      assert_equal(err, "")
      set_lang(C)
      out, err = capture_subprocess_io do
        decl Int(:a)
      end
      assert_equal(out, "int64_t a;\n")
      assert_equal(err, "")
    ensure
      pop_env(:default_int_size)
    end
  end

  def test_decl_int_array
    set_lang(FORTRAN)
    n1 = Int("n1")
    arr = Int(:a, :dim => [Dim(5,15), Dim(n1)])
    out, err = capture_subprocess_io do
      decl arr
    end
    assert_equal(out, "integer(kind=4), dimension(5:15, n1) :: a\n")
    assert_equal(err, "")
    set_lang(C)
    out, err = capture_subprocess_io do
      decl arr
    end
    assert_equal(out, "int32_t * a;\n")
    assert_equal(err, "")
    push_env(:use_vla => true)
    out, err = capture_subprocess_io do
      decl arr
    end
    assert_equal(out, "int32_t a[n1][11];\n")
    assert_equal(err, "")
    pop_env(:use_vla)
  end

  def test_decl_int_array_unkwown_dim
    set_lang(FORTRAN)
    arr = Int(:a, :dim => [Dim(5,15), Dim()])
    out, err = capture_subprocess_io do
      decl arr
    end
    assert_equal(out, "integer(kind=4), dimension(5:15, *) :: a\n")
    assert_equal(err, "")
    set_lang(C)
    out, err = capture_subprocess_io do
      decl arr
    end
    assert_equal(out, "int32_t * a;\n")
    assert_equal(err, "")
    begin
      push_env(:use_vla => true)
      out, err = capture_subprocess_io do
        decl arr
      end
      assert_equal(out, "int32_t a[][11];\n")
      assert_equal(err, "")
    ensure
      pop_env(:use_vla)
    end
  end

  def test_decl_int_array_deffered_shape
    set_lang(FORTRAN)
    arr = Int(:a, :dim => [Dim(5,15), Dim()], :deferred_shape => true)
    out, err = capture_subprocess_io do
      decl arr
    end
    assert_equal(out, "integer(kind=4), dimension(:, :) :: a\n")
    assert_equal(err, "")
  end

  def test_pr_int_array_index
    set_lang(FORTRAN)
    n1 = Int("n1")
    arr = Int(:a, :dim => [Dim(5,15), Dim(n1)])
    out, err = capture_subprocess_io do
      pr arr[6,7]
    end
    assert_equal(out, "a(6, 7)\n")
    assert_equal(err, "")
    set_lang(C)
    out, err = capture_subprocess_io do
      pr arr[6,7]
    end
    assert_equal(out, "a[6 - (5) + (11) * (7 - (1))];\n")
    assert_equal(err, "")
    begin
      push_env(:use_vla => true)
      out, err = capture_subprocess_io do
        pr arr[6,7]
      end
      assert_equal(out, "a[7 - (1)][6 - (5)];\n")
      assert_equal(err, "")
    ensure
      pop_env(:use_vla)
    end
  end

  def test_pr_int_array_index_unkwown_dim
    n1 = Int("n1")
    set_lang(FORTRAN)
    arr = Int(:a, :dim => [Dim(n1), Dim()])
    out, err = capture_subprocess_io do
      pr arr[6,7]
    end
    assert_equal(out, "a(6, 7)\n")
    assert_equal(err, "")
    set_lang(C)
    out, err = capture_subprocess_io do
      pr arr[6,7]
    end
    assert_equal(out, "a[6 - (1) + (n1) * (7 - (1))];\n")
    assert_equal(err, "")
    begin
      push_env(:use_vla => true, :array_start => 0)
      out, err = capture_subprocess_io do
        pr arr[6,7]
      end
      assert_equal(out, "a[7][6];\n")
      assert_equal(err, "")
    ensure
      pop_env(:use_vla, :array_start)
    end
  end

end
