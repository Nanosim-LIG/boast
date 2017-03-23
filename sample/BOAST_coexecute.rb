[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'BOAST'
include BOAST

set_lang C

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
ah1 = NArray.int(2**22)
ah2 = NArray.int(2**22)

check = true

opts = { repeat: 1..10, size: 10..22 }
opt_space = OptimizationSpace::new(opts)
optimizer = BruteForceOptimizer::new( opt_space )

k1 = p.ckernel
k2 = p.ckernel

k1.build
k2.build

optimizer.each { |config|
  p config
  ah1.random!(100)
  ah2.random!(100)

  repeat = 2**config[:repeat]
  size = 2**config[:size]

  if check then
    a1_out_ref = ah1 + 3 * (repeat*2)
    a2_out_ref = ah2 + 2 * (repeat*2)
  end

  res1 = k1.run( size, ah1, 3, :repeat => repeat, :cpu_affinity => [0])
  res2 = k2.run( size, ah2, 2, :repeat => repeat, :cpu_affinity => [1])

  puts "sequential: #{res1[:duration]} + #{res2[:duration]} = #{res1[:duration]+res2[:duration]}"

  res = CKernel::coexecute([ [k1, [size, ah1, 3, {:cpu_affinity => [0], :repeat => repeat}]], [k2, [size, ah2, 2,{:cpu_affinity => [1], :repeat => repeat}]] ])
  dates = [[0, :start, res[0][:start]], [1, :start, res[1][:start]], [0, :end, res[0][:end]], [1, :end, res[1][:end]]]
  dates.sort! { |e1,e2| e1[2] <=> e2[2] }
#  p dates
  duration = (dates[-1][2] - dates[0][2])/1e9
  if dates[0][1] == :start and dates[1][1] == :start then
    overlap = (dates[2][2] - dates[1][2]) /(1e9 * duration)
  else
    overlap = 0
  end

  puts "parallel: #{(dates[1][2]-dates[0][2])/1e9} + #{(dates[2][2] - dates[1][2])/1e9} + #{(dates[3][2]- dates[2][2])/1e9} = #{duration} (overlap: #{overlap*100}%)"
  puts "speedup: #{(res1[:duration]+res2[:duration])/duration}"

  if check then
    raise "Computation error!" unless a1_out_ref[0...size] == ah1[0...size]
    raise "Computation error!" unless a2_out_ref[0...size] == ah2[0...size]
  end
}
