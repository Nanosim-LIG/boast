require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestArray < Minitest::Test

  def test_decl_int_array
    n1 = Int("n1")
    arr = Int(:a, :dim => [Dim(5,15), Dim(n1)])
    block = lambda { decl arr }
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4), dimension(5:15, n1) :: a\n", "", &block )
    set_lang(C)
    assert_subprocess_output( "int32_t * a;\n", "", &block )
    set_lang(CUDA)
    assert_subprocess_output( "int * a;\n", "", &block )
    set_lang(CL)
    assert_subprocess_output( "int * a;\n", "", &block )
    begin
      push_env(:use_vla => true)
      set_lang(C)
      assert_subprocess_output( "int32_t a[n1][11];\n", "", &block )
      set_lang(CUDA)
      assert_subprocess_output( "int * a;\n", "", &block )
      set_lang(CL)
      assert_subprocess_output( "int * a;\n", "", &block )
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
    set_lang(CUDA)
    assert_subprocess_output( "int * a;\n", "", &block )
    set_lang(CL)
    assert_subprocess_output( "int * a;\n", "", &block )
    begin
      push_env(:use_vla => true)
      set_lang(C)
      assert_subprocess_output( "int32_t a[][11];\n", "", &block )
      set_lang(CUDA)
      assert_subprocess_output( "int * a;\n", "", &block )
      set_lang(CL)
      assert_subprocess_output( "int * a;\n", "", &block )
    ensure
      pop_env(:use_vla)
    end
  end

  def test_decl_int_array_deffered_shape
    arr = Int(:a, :dim => [Dim(5,15), Dim()], :deferred_shape => true)
    block = lambda { decl arr }
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4), dimension(:, :) :: a\n", "", &block )
    set_lang(C)
    assert_subprocess_output( "int32_t * a;\n", "", &block )
    set_lang(CUDA)
    assert_subprocess_output( "int * a;\n", "", &block )
    set_lang(CL)
    assert_subprocess_output( "int * a;\n", "", &block )
  end

  def test_pr_int_array_index
    n1 = Int("n1")
    arr = Int(:a, :dim => [Dim(5,15), Dim(n1)])
    block = lambda { pr arr[6,7] }
    set_lang(FORTRAN)
    assert_subprocess_output( "a(6, 7)\n", "", &block )
    [C, CL, CUDA].each { |l|
      set_lang(l)
      assert_subprocess_output( "a[6 - (5) + (11) * (7 - (1))];\n", "", &block )
    }
    begin
      push_env(:use_vla => true)
      set_lang(C)
      assert_subprocess_output( "a[7 - (1)][6 - (5)];\n", "", &block )
      set_lang(CUDA)
      assert_subprocess_output( "a[6 - (5) + (11) * (7 - (1))];\n", "", &block )
      set_lang(CL)
      assert_subprocess_output( "a[6 - (5) + (11) * (7 - (1))];\n", "", &block )
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
    [C, CL, CUDA].each { |l|
      set_lang(l)
      assert_subprocess_output( "a[6 - (1) + (n1) * (7 - (1))];\n", "", &block )
    }
    begin
      push_env(:use_vla => true, :array_start => 0)
      set_lang(C)
      assert_subprocess_output( "a[7][6];\n", "", &block )
      set_lang(CL)
      assert_subprocess_output( "a[6 + (n1) * (7)];\n", "", &block )
      set_lang(CUDA)
      assert_subprocess_output( "a[6 + (n1) * (7)];\n", "", &block )
    ensure
      pop_env(:use_vla, :array_start)
    end
  end

  def test_decl_int_array_local
    n1 = Int("n1")
    arr = Int(:a, :dim => [Dim(5,15), Dim(n1)], :local => true)
    block = lambda { decl arr }
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4), dimension(5:15, n1) :: a\n", "", &block )
    set_lang(C)
    assert_subprocess_output( "int32_t a[(n1)*(11)];\n", "", &block )
    set_lang(CL)
    assert_subprocess_output( "__local int a[(n1)*(11)];\n", "", &block )
    set_lang(CUDA)
    assert_subprocess_output( "__shared__ int a[(n1)*(11)];\n", "", &block )
  end

  def test_decl_int_array_allocate
    n1 = Int("n1")
    arr = Int(:a, :dim => [Dim(5,15), Dim(n1)], :allocate => :heap)
    block = lambda { decl arr }
    block2 = lambda { pr arr.alloc }
    block3 = lambda { pr arr.dealloc }
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4), allocatable, dimension(:, :) :: a\n", "", &block )
    assert_subprocess_output( "allocate(a(5:15, n1))\n", "", &block2 )
    assert_subprocess_output( "deallocate(a)\n", "", &block3 )
    set_lang(C)
    assert_subprocess_output( "int32_t * a;\n", "", &block )
    assert_subprocess_output( "a = (int32_t *)malloc((sizeof(int32_t)) * (n1)*(11));\n", "", &block2 )
    assert_subprocess_output( "free(a);\n", "", &block3 )
  end

  def test_decl_int_array_allocate
    n1 = Int("n1")
    arr = Int(:a, :dim => [Dim(5,15), Dim(n1)], :allocate => :heap, :align => 32)
    block = lambda { decl arr }
    block2 = lambda { pr arr.alloc(nil, 32) }
    block3 = lambda { pr arr.dealloc }
    set_lang(FORTRAN)
    assert_subprocess_output( "integer(kind=4), allocatable, dimension(:, :) :: a\n", "", &block )
    assert_subprocess_output( "allocate(a(5:15, n1))\n", "", &block2 )
    assert_subprocess_output( "deallocate(a)\n", "", &block3 )
    set_lang(C)
    assert_subprocess_output( "int32_t * a;\n", "", &block )
    assert_subprocess_output( "posix_memalign( &a, 32, (sizeof(int32_t)) * (n1)*(11));\n", "", &block2 )
    assert_subprocess_output( "free(a);\n", "", &block3 )
  end

end
