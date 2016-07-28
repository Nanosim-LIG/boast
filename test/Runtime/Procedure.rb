require 'minitest/autorun'
require 'BOAST'
require 'narray_ffi'
include BOAST
require_relative '../helper'

def silence_warnings(&block)
  warn_level = $VERBOSE
  $VERBOSE = nil
  result = block.call
  $VERBOSE = warn_level
  result
end


class TestProcedure < Minitest::Test

  def test_repeat
    a = Int(:a, :dim => Dim(1), :dir => :inout)
    p = Procedure("inc", [a]) { pr a[1] === a[1] + 1 }
    [FORTRAN, C].each { |l|
      ah = NArray::int(1).fill!(0)
      set_lang(l)
      k = p.ckernel
      k.run(ah, :repeat => 15)
      assert_equal(15, ah[0])
    }
  end

  def test_procedure
    a = Int( :a, :dir => :in )
    b = Int( :b, :dir => :in )
    c = Int( :c, :dir => :out )
    p = Procedure("minimum", [a,b,c]) { pr c === Ternary( a < b, a, b) }
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
    p = Procedure("minimum", [a,b], :return => c) { pr c === Ternary( a < b, a, b) }
    [FORTRAN, C].each { |l|
      set_lang(l)
      k = p.ckernel
      r = k.run(10, 5)
      assert_equal(5, r[:return])
    }
  end

  def test_procedure_array
    n = Int( :n, :dir => :in )
    a = Int( :a, :dir => :inout, :dim => [Dim(n)] )
    b = Int( :b, :dir => :in )
    i = Int( :i )
    p = Procedure("vector_inc", [n, a, b]) {
      decl i
      pr For(i, 1, n) {
        pr a[i] === a[i] + b
      }
    }
    ah = NArray.int(1024)
    [FORTRAN, C].each { |l|
      set_lang(l)
      ah.random!(100)
      a_out_ref = ah + 2
      k = p.ckernel
      r = k.run(ah.size, ah, 2)
      assert_equal(a_out_ref, ah)
    }
  end

  def test_procedure_opencl_array
    begin
      silence_warnings { require 'opencl_ruby_ffi' }
    rescue
      skip "Missing OpenCL on the platform!"
    end
    push_env(:array_start => 0) {
      a = Int( :a, :dir => :inout, :dim => [Dim()] )
      b = Int( :b, :dir => :in )
      i = Int( :i )
      p = Procedure("vector_inc", [a, b]) {
        decl i
        pr i === get_global_id(0)
        pr a[i] === a[i] + b
      }
      nelem = 1024
      ah = NArray.int(nelem)
      set_lang(CL)
      ah.random!(100)
      a_out_ref = ah + 2
      k = p.ckernel
      r = k.run(ah, 2, :global_work_size => [nelem,1,1], :local_work_size => [32,1,1])
      assert_equal(a_out_ref, ah)
    }
  end

end
