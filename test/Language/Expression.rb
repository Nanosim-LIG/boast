require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestExpression < Minitest::Test

  def test_arithmetic
    exp = lambda { |a,b,c,d,e,f,g,h|
      a + +b + -(c * d)/e +g*h
    }
    vals = 8.times.collect { rand(100) + 1 }
    vals_var = vals.collect(&:to_var)
    puts exp.call(*vals_var)
    assert_equal( exp.call(*vals), eval(exp.call(*vals_var).to_s) )
  end

end
