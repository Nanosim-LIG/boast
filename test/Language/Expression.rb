require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

def pow(a, b)
  return a**b
end

class TestExpression < Minitest::Test

  def test_arithmetic
    exp = lambda { |a,b,c,d,e,f,g,h|
      a + +b + -(c * d)/e +g**h
    }
    vals = 8.times.collect { rand(100) + 1 }
    vals_var = vals.collect(&:to_var)
    [FORTRAN,C,CL,CUDA].each { |l|
      set_lang(l)
      assert_equal( exp.call(*vals), eval(exp.call(*vals_var).to_s) )
    }
  end

  def test_logic
    ruby_exp = lambda { |a,b,c,d,e,f,g,h|
      a < b || b >= c + a && !(d - e == g + h)
    }
    boast_exp = lambda { |a,b,c,d,e,f,g,h|
      (a < b) | ( (b >= c + a) &  (!(d - e == g + h) ) )
    }
    boast_exp2 = lambda { |a,b,c,d,e,f,g,h|
      (a < b).or( (b >= c + a).and(!(d - e == g + h) ) )
    }
    boast_exp3 = lambda { |a,b,c,d,e,f,g,h|
      Or( a < b, And( b >= c + a, !(d - e == g + h) ) )
    }
    vals = 8.times.collect { rand(10) }
    vals_var = vals.collect(&:to_var)
    set_lang(FORTRAN)
    assert_equal( ruby_exp.call(*vals), eval( boast_exp.call(*vals_var).to_s.gsub(".and.","&&").gsub(".or.","||").gsub(".not.", "!")))
    assert_equal( ruby_exp.call(*vals), eval(boast_exp2.call(*vals_var).to_s.gsub(".and.","&&").gsub(".or.","||").gsub(".not.", "!")))
    assert_equal( ruby_exp.call(*vals), eval(boast_exp3.call(*vals_var).to_s.gsub(".and.","&&").gsub(".or.","||").gsub(".not.", "!")))
    [C,CL,CUDA].each { |l|
      set_lang(l)
      assert_equal( ruby_exp.call(*vals), eval( boast_exp.call(*vals_var).to_s) )
      assert_equal( ruby_exp.call(*vals), eval(boast_exp2.call(*vals_var).to_s) )
      assert_equal( ruby_exp.call(*vals), eval(boast_exp3.call(*vals_var).to_s) )
    }
  end

end
