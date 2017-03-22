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
    p = Procedure("minimum_1", [a,b,c]) { pr c === Ternary( a < b, a, b) }
    [FORTRAN, C].each { |l|
      set_lang(l)
      k = p.ckernel
      r = k.run(10, 5, 0)
      assert_equal(5, r[:reference_return][:c])
    }
  end

  def test_procedure_reference
    a = Int( :a, :dir => :in )
    b = Int( :b, :dir => :in, :reference => true )
    c = Int( :c, :dir => :out )
    p = Procedure("minimum_2", [a,b,c]) { pr c === Ternary( a < b, a, b) }
    [FORTRAN, C].each { |l|
      set_lang(l)
      k = p.ckernel
      r = k.run(10, 5, 0)
      assert_equal(5, r[:reference_return][:c])
      assert_equal({:c=>5}, r[:reference_return])
    }
  end

  def test_procedure_vector
    skip if get_architecture == ARM
    b = Real( :b, :dir => :in, :vector_length => 2 )
    c = Real( :c, :dir => :out, :vector_length => 2, :dim => Dim(4) )
    p = Procedure("vector_copy", [b,c]) { pr c[1] === b }
    b_a = ANArray.float( 16, 2 ).random!
    c_a = ANArray.float( 16, 2, 4 )
    [FORTRAN, C].each { |l|
      c_a.random!
      set_lang(l)
      k = p.ckernel( :includes => "immintrin.h")
      k.run(b_a, c_a)
      assert_equal(0.0, (b_a[0..1] - c_a[0..1]).abs.max)
    }
  end

  def test_function
    a = Int( :a, :dir => :in )
    b = Int( :b, :dir => :in )
    c = Int( :c )
    p = Procedure("minimum_3", [a,b], :return => c) { pr c === Ternary( a < b, a, b) }
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
    p = Procedure("vector_inc_1", [n, a, b]) {
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
      k.run(ah.size, ah, 2)
      assert_equal(a_out_ref, ah)
    }
  end

  def test_coexecute
    a1 = Int( :a, :dir => :in )
    b1 = Int( :b, :dir => :in )
    c1 = Int( :c )
    p1 = Procedure("minimum_4", [a1,b1], :return => c1) {
      pr c1 === Ternary( a1 < b1, a1, b1)
    }
    k1 = p1.ckernel

    n2 = Int( :n, :dir => :in )
    a2 = Int( :a, :dir => :inout, :dim => [Dim(n2)] )
    b2 = Int( :b, :dir => :in )
    i2 = Int( :i )
    p2 = Procedure("vector_inc_2", [n2, a2, b2]) {
      decl i2
      pr For(i2, 1, n2) {
        pr a2[i2] === a2[i2] + b2
      }
    }
    k2 = p2.ckernel

    a2h = NArray.int(1024)
    [FORTRAN, C].each { |l|
      set_lang(l)
      k1 = p1.ckernel
      k2 = p2.ckernel
      a2h.random!(100)
      a2_out_ref = a2h + 2
      res = CKernel::coexecute([ [k1, [10, 5]], [k2, [a2h.size, a2h, 2]] ])
      assert_equal(5, res[0][:return])
      assert_equal(a2_out_ref, a2h)
    }
  end

  def test_coexecute2
    n = Int( :n, :dir => :in )
    a = Int( :a, :dir => :inout, :dim => [Dim(n)] )
    b = Int( :b, :dir => :in )
    i = Int( :i )
    p = Procedure("vector_inc_3", [n, a, b]) {
      decl i
      pr For(i, 1, n) {
        pr a[i] === a[i] + b
      }
    }
    ah1 = NArray.int(1024*1024)
    ah2 = NArray.int(1024*1024)
    [FORTRAN, C].each { |l|
      set_lang(l)
      k1 = p.ckernel
      k2 = p.ckernel
      ah2.random!(100)
      ah1.random!(100)
      a1_out_ref = ah1 + 3*2
      a2_out_ref = ah2 + 2*2
      CKernel::coexecute([ [k1, [ah1.size, ah1, 3]], [k2, [ah2.size, ah2, 2]] ])
      res = CKernel::coexecute([ [k1, [ah1.size, ah1, 3]], [k2, [ah2.size, ah2, 2]] ])
      assert_equal(a2_out_ref, ah2)
      assert_equal(a1_out_ref, ah1)
      assert( res[1][:start] < res[0][:end] && res[0][:start] < res[1][:end] ) unless executable? or ffi?
    }
  end

  def test_coexecute3
    n = Int( :n, :dir => :in )
    a = Int( :a, :dir => :inout, :dim => [Dim(n)] )
    b = Int( :b, :dir => :in )
    i = Int( :i )
    p = Procedure("vector_inc_4", [n, a, b]) {
      decl i
      pr For(i, 1, n) {
        pr a[i] === a[i] + b
      }
    }
    ah1 = NArray.int(1024*1024)
    ah2 = NArray.int(1024*1024)
    [FORTRAN, C].each { |l|
      set_lang(l)
      repeat = 2
      k1 = p.ckernel
      k2 = p.ckernel
      ah2.random!(100)
      ah1.random!(100)
      a1_out_ref = ah1 + 3 * (repeat)
      a2_out_ref = ah2 + 2 * (repeat)
      k1.run( ah1.size, ah1, 3, :repeat => repeat/2, :cpu_affinity => [0])
      k2.run( ah2.size, ah2, 2, :repeat => repeat/2, :cpu_affinity => [3])
      res = CKernel::coexecute([ [k1, [ah1.size, ah1, 3, {:cpu_affinity => [1], :repeat => repeat/2}]], [k2, [ah2.size, ah2, 2,{:cpu_affinity => [2], :repeat => repeat/2}]] ])
      assert_equal(a2_out_ref, ah2)
      assert_equal(a1_out_ref, ah1)
      assert( res[1][:start] < res[0][:end] && res[0][:start] < res[1][:end] ) unless executable? or ffi?
    }
  end

  def test_procedure_opencl_array
    begin
      silence_warnings { require 'opencl_ruby_ffi' }
      plts = OpenCL::platforms
    rescue
      skip "Missing OpenCL on the platform!"
    end
    push_env(:array_start => 0) {
      a = Int( :a, :dir => :inout, :dim => [Dim()] )
      b = Int( :b, :dir => :in )
      i = Int( :i )
      p = Procedure("vector_inc_2", [a, b]) {
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
      k.run(ah, 2, :global_work_size => [nelem,1,1], :local_work_size => [32,1,1])
      assert_equal(a_out_ref, ah)
    }
  end

  def test_procedure_opencl_array_repeat
    begin
      silence_warnings { require 'opencl_ruby_ffi' }
      plts = OpenCL::platforms
    rescue
      skip "Missing OpenCL on the platform!"
    end
    push_env(:array_start => 0) {
      repeat = 3
      a = Int( :a, :dir => :inout, :dim => [Dim()] )
      b = Int( :b, :dir => :in )
      i = Int( :i )
      p = Procedure("vector_inc_3", [a, b]) {
        decl i
        pr i === get_global_id(0)
        pr a[i] === a[i] + b
      }
      nelem = 1024
      ah = NArray.int(nelem)
      set_lang(CL)
      ah.random!(100)
      a_out_ref = ah + 2 * repeat
      k = p.ckernel
      k.run(ah, 2, :global_work_size => [nelem,1,1], :local_work_size => [32,1,1], :repeat => repeat)
      assert_equal(a_out_ref, ah)
    }
  end

end
