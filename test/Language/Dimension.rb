require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestDimension < Minitest::Test

  def test_dim_1_param
    d = Dim(15)
    assert_equal("15", "#{d.size}")
    assert_equal("1" , "#{d.start}")
    assert_equal("15", "#{d.finish}")
  end

  def test_dim_1_param_array_start_0
    push_env(:array_start => 0) {
      d = Dim(15)
      assert_equal("15", "#{d.size}")
      assert_equal("0" , "#{d.start}")
      assert_equal("14", "#{d.finish}")
    }
  end

  def test_dim_1_param_array_start_2
    push_env(:array_start => 2) {
      d = Dim(15)
      assert_equal("15", "#{d.size}")
      assert_equal("2" , "#{d.start}")
      assert_equal("16", "#{d.finish}")
    }
  end

  def test_dim_1_param_array_start_expr
    push_env(:array_start => Int(:n) - 1) {
      d = Dim(15)
      assert_equal("15", "#{d.size}")
      assert_equal("n - (1)", "#{d.start}")
      assert_equal("15 + n - (1) - (1)", "#{d.finish}")
    }
  end

  def test_dim_1_expr_param
    expr = Int(:n) - 1
    d = Dim(expr)
    assert_equal("n - (1)", "#{d.size}")
    assert_equal("1", "#{d.start}")
    assert_equal("n - (1)", "#{d.finish}")
  end

  def test_dim_1_expr_param_array_start_0
    push_env(:array_start => 0) {
      expr = Int(:n) - 1
      d = Dim(expr)
      assert_equal("n - (1)", "#{d.size}")
      assert_equal("0", "#{d.start}")
      assert_equal("n - (1) - (1)", "#{d.finish}")
    }
  end

  def test_dim_1_expr_param_array_start_2
    push_env(:array_start => 2) {
      expr = Int(:n) - 1
      d = Dim(expr)
      assert_equal("n - (1)", "#{d.size}")
      assert_equal("2", "#{d.start}")
      assert_equal("n - (1) + 2 - (1)", "#{d.finish}")
    }
  end

  def test_dim_1_expr_param_array_start_expr
    push_env(:array_start => Int(:m) + 2) {
      d = Dim( Int(:n) - 1 )
      assert_equal("n - (1)", "#{d.size}")
      assert_equal("m + 2", "#{d.start}")
      assert_equal("n - (1) + m + 2 - (1)", "#{d.finish}")
    }
  end

  def test_dim_2_params
    d = Dim(5,15)
    assert_equal("11", "#{d.size}")
    assert_equal("5" , "#{d.start}")
    assert_equal("15", "#{d.finish}")
  end

  def test_dim_2_expr_params
    expr = Int(:n) - 1
    d = Dim(5,expr)
    assert_equal("n - (1) - (5) + 1", "#{d.size}")
    assert_equal("5", "#{d.start}")
    assert_equal("n - (1)", "#{d.finish}")
    d = Dim(expr,15)
    assert_equal("15 - (n - (1)) + 1", "#{d.size}")
    assert_equal("n - (1)", "#{d.start}")
    assert_equal("15", "#{d.finish}")
    expr2 = Int(:m) + 2
    d = Dim(expr2,expr)
    assert_equal("n - (1) - (m + 2) + 1", "#{d.size}")
    assert_equal("m + 2", "#{d.start}")
    assert_equal("n - (1)", "#{d.finish}")
  end

  def test_dim_2_param_array_start_0
    push_env(:array_start => 0) {
      d = Dim(5,15)
      assert_equal("11", "#{d.size}")
      assert_equal("5" , "#{d.start}")
      assert_equal("15", "#{d.finish}")
    }
  end

  def test_dim_2_param_array_start_2
    push_env(:array_start => 2) {
      d = Dim(5,15)
      assert_equal("11", "#{d.size}")
      assert_equal("5" , "#{d.start}")
      assert_equal("15", "#{d.finish}")
    }
  end

end
