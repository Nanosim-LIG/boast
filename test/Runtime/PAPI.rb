require 'minitest/autorun'
require 'BOAST'
require 'narray_ffi'
include BOAST
require_relative '../helper'

class TestPAPI < Minitest::Test

  def test_flops
    size = 1024*1024
    a = Real(:a, :dim => Dim(size), :dir => :inout)
    p = Procedure("inc", [a]) {
      decl i = Int(:i)
      pr For(i,1,size) {
        pr a[i] === a[i] + 1.0
      }
    }
    [FORTRAN, C].each { |l|
      ah = NArray::float(size).random!
      set_lang(l)
      k = p.ckernel
      stats = k.run(ah, :PAPI =>["PAPI_DP_OPS"] )
      assert_equal(size, stats[:PAPI]["PAPI_DP_OPS"] )
    }
    PAPI.shutdown
  end

  def test_omp
    size = 1024*1024
    a = Real(:a, :dim => Dim(size), :dir => :inout)
    p = Procedure("inc", [a]) {
      decl i = Int(:i)
      pr OpenMP::Parallel(:private => i, :shared => a) {
        pr For(i,1,size, :openmp => true) {
          pr a[i] === a[i] + 1.0
        }
      }
    }
    [FORTRAN, C].each { |l|
      ah = NArray::float(size).random!
      set_lang(l)
      k = p.ckernel
      k.build(:openmp => true)
      stats = k.run(ah, :PAPI =>["PAPI_DP_OPS"] )
      assert(size >= stats[:PAPI]["PAPI_DP_OPS"] )
    }
    PAPI.shutdown
  end

  def test_coexecute
    size = 1024*1024
    s = Int(:s)
    a = Real(:a, :dim => Dim(size), :dir => :inout)
    p = Procedure("inc", [s,a]) {
      decl i = Int(:i)
      pr For(i,1,s) {
        pr a[i] === a[i] + 1.0
      }
    }
    [FORTRAN, C].each { |l|
      ah1 = NArray::float(2*size).random!
      ah2 = NArray::float(size).random!
      set_lang(l)
      k1 = p.ckernel
      k2 = p.ckernel
      res = CKernel.coexecute( [ [k1, [2*size, ah1, :PAPI =>["PAPI_DP_OPS"]] ], [k2, [size, ah2, :PAPI =>["PAPI_DP_OPS"]] ] ] )
      assert_equal(2*size, res[0][:PAPI]["PAPI_DP_OPS"] )
      assert_equal(size, res[1][:PAPI]["PAPI_DP_OPS"] )
    }
    PAPI.shutdown
  end

end
