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

  def test_powercap
    n = Int( :n, :dir => :in )
    a = Int( :a, :dir => :in, :dim => [Dim(n)] )
    b = Int( :b, :dir => :in, :dim => [Dim(n)] )
    c = Int( :c, :dir => :inout, :dim => [Dim(n)] )
    i = Int( :i )
    j = Int( :j )
    p = Procedure("vector_inc", [n, a, b, c]) {
      decl i
      decl j
      pr For(j, 1, 32*1024) {
        pr For(i, 1, n) {
          pr c[i] === a[i] * b[i]
        }
      }
    }
    r = []
    n = 1024
    ah = NArray.int(n)
    bh = NArray.int(n)
    ch = NArray.int(n)
    set_lang(C)
    ah.random!(n)
    bh.random!(n)
    ch.random!(n)
    k = p.ckernel
    8.times { k.run(n, ah, bh, ch) }
    r = k.run(n, ah, bh, ch)
    t0 = r[:duration]
    e0 = r[:energy]
    8.times { k.run(n, ah, bh, ch) }
    r = k.run(n, ah, bh, ch)
    t1 = r[:duration]
    e1 = r[:energy]
    e0.each {|name, x|
      energy0 = e0[name.to_sym]
      energy1 = e1[name.to_sym]
      next if energy0 < 0.01 and energy1 < 0.01
      assert(((energy0 / t0 - energy1 / t1).abs) / (energy1 / t1) < 0.05)
    }
  end

end
