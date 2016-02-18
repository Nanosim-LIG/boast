require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestProcedure < Minitest::Test

  def test_procedure
    a = Int( :a, :dir => :in )
    b = Int( :b, :dir => :in )
    c = Int( :c, :dir => :out )
    p = Procedure("minimum", [a,b,c]) { pr c === Ternary( a < b, a, b) }
    block = lambda { pr p }
    [FORTRAN, C].each { |l|
      set_lang(l)
      k = p.ckernel
      r = k.run(10, 5, 0)
      assert_equal(5, r[:reference_return][:c])
    }
  end

  def test_function
    a = Int( :a, :dir => :in )
    b = Int( :b, :dir => :in )
    c = Int( :c )
    p = Procedure("minimum", [a,b], [], :return => c) { pr c === Ternary( a < b, a, b) }
    block = lambda { pr p }
    [FORTRAN, C].each { |l|
      set_lang(l)
      k = p.ckernel
      r = k.run(10, 5)
      assert_equal(5, r[:return])
    }
  end

end
