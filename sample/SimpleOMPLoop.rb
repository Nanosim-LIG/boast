[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'BOAST'
include BOAST


def simple_loop(omp = false)
  n = Int :n, dir: :in
  a = Real :a, dim: [Dim(n)], dir: :in
  c = Real :c
  p = Procedure(:test_loop, [n, a], return: c) {
    i = Int :i
    j = Int :j
    t = Real :t
    decl i, j, t
    pr c === 0.0
    f = For(i, 1, n) {
      pr t === a[i]
      pr For(j, 1, 1000) {
        pr t === t * 1.0001 + 0.0001
      }
      pr c === c + t
    }
    if omp
      pr OpenMP::ParallelFor( private: [i, j, t], shared: [a, n], reduction: {"+" => c } ) {
        pr f
      }
    else
      pr f
    end
  }
  return p.ckernel
end

n = 1024*1024
a = NArray.float(n).random!


c_ref = 0.0
a.each { |e|
  t = e
  1000.times {
    t = t * 1.0001 + 0.0001
  }
  c_ref += t
}


[false, true].each { |omp|

  k = simple_loop(omp)
  puts k

  k.build( openmp: omp )
  stats = k.run(n, a)
  stats = k.run(n, a)
  stats = k.run(n, a, PAPI: [ 'PAPI_TOT_INS', 'PAPI_TOT_CYC'])
  c = stats[:return]

  raise "Compute error: #{ (c_ref - c).abs } !" if (c_ref - c).abs > 10e-6
  puts "parallel: #{omp}"
  puts stats

}
