require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

def pow(a, b)
  return a**b
end

class TestExpression < Minitest::Test

  def test_replace_constant
    a = 5
    push_env( :lang => C, :replace_constants => true ) {
      i = Int :i, :const => 5
      block = lambda { pr i - 1 }
assert_subprocess_output( <<EOF, "", &block )
5 - (1);
EOF
    }
  end

  def test_empty_return
    block = lambda { pr Return(nil) }
    set_lang(FORTRAN)
    assert_subprocess_output( " return \n", "", &block )
    [C,CL,CUDA,HIP].each { |l|
      set_lang(l)
      assert_subprocess_output( <<EOF, "", &block )
 return ;
EOF
    }
  end

  def test_coerce
    i = Int :i
    a = Real :a
    block = lambda { pr a === 2.0 * a + 5 * i }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
a = (2.0_wp) * (a) + (5) * (i)
EOF
    [C,CL,CUDA,HIP].each { |l|
      set_lang(l)
      assert_subprocess_output( <<EOF, "", &block )
a = (2.0) * (a) + (5) * (i);
EOF
    }
  end

  def test_arithmetic
    exp = lambda { |a,b,c,d,e,f,g,h|
      a + +b + -(c * d)/e + g**h
    }
    vals = 8.times.collect { rand(100) + 1 }
    vals_var = vals.collect(&:to_var)
    [FORTRAN,C,CL,CUDA,HIP].each { |l|
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
    [C,CL,CUDA,HIP].each { |l|
      set_lang(l)
      assert_equal( ruby_exp.call(*vals), eval( boast_exp.call(*vals_var).to_s) )
      assert_equal( ruby_exp.call(*vals), eval(boast_exp2.call(*vals_var).to_s) )
      assert_equal( ruby_exp.call(*vals), eval(boast_exp3.call(*vals_var).to_s) )
    }
  end

  def test_fma
    a = Int :a
    b = Int :b
    c = Int :c
    block = lambda { pr c === FMA(a,b,c) }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
c = c + (a) * (b)
EOF
    set_lang(C)
    assert_subprocess_output( <<EOF, "", &block )
c = c + (a) * (b);
EOF
    set_lang(CL)
    assert_subprocess_output( <<EOF, "", &block )
c = fma( a, b, c );
EOF
    set_lang(CUDA)
    assert_subprocess_output( <<EOF, "", &block )
c = fma( a, b, c );
EOF
    set_lang(HIP)
    assert_subprocess_output( <<EOF, "", &block )
c = fma( a, b, c );
EOF

  end

  def test_fms
    a = Int :a
    b = Int :b
    c = Int :c
    block = lambda { pr c === FMS(a,b,c) }
    set_lang(FORTRAN)
    assert_subprocess_output( <<EOF, "", &block )
c = c - ((a) * (b))
EOF
    set_lang(C)
    assert_subprocess_output( <<EOF, "", &block )
c = c - ((a) * (b));
EOF
    set_lang(CL)
    assert_subprocess_output( <<EOF, "", &block )
c = fma(  -(a), b, c );
EOF
    set_lang(CUDA)
    assert_subprocess_output( <<EOF, "", &block )
c = fma(  -(a), b, c );
EOF
    set_lang(HIP)
    assert_subprocess_output( <<EOF, "", &block )
c = fma(  -(a), b, c );
EOF
  end

end
