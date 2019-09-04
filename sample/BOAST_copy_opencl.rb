require 'BOAST'
require 'narray_ffi'

include BOAST

real_size = 4
BOAST.set_array_start(0)
BOAST.set_default_real_size(real_size)

narra_types = {
  8 => NArray::FLOAT,
  4 => NArray::SFLOAT
}


BOAST.set_lang( BOAST::CL )
def kernel_copy(vector_length: 2, block_size_x: 1)
  function_name = "copy_kern"
  a = Real :a, vector_length: vector_length, dir: :in, dim: [ Dim(block_size_x), Dim() ]
  b = Real :b, vector_length: vector_length, dir: :out, dim: [ Dim(block_size_x), Dim() ]

  i = Int :i, signed: false

  args = [a, b]

  p = Procedure( function_name, args ) {
    decl i
    pr i === get_global_id(0)
    block_size_x.times { |j|
      pr b[j, i] === a[j, i]
    }
  }

  p.ckernel
end

n = 1<<28
n = 12_800_000/(real_size/4)

a = ANArray.new(narra_types[real_size],32, n).random!
b = ANArray.new(narra_types[real_size],32, n).random!

vector_lengths = [1,2]
vector_lengths.push 4 if real_size == 4

opt_space = OptimizationSpace::new(
  vector_length: vector_lengths,
  local_work_size: [8, 16, 32, 64, 128, 256],
  block_size_x: [1, 2, 4]
)

optimizer = BruteForceOptimizer::new(opt_space, :randomize => true)

repeat = 100

config, time = optimizer.optimize { |opt|
  puts opt
  vector_length = opt[:vector_length]
  local_work_size = opt[:local_work_size]
  block_size_x = opt[:block_size_x]
  

  a.random!
  b.random!
  k = kernel_copy(vector_length: vector_length, block_size_x: block_size_x)

  puts k
  res = k.run(a, b, global_work_size: [n/(vector_length*block_size_x)], local_work_size: [local_work_size] )
  err = b - a
  puts "Error: #{err.abs.max}"
 
  results = 5.times.collect { 
    k.run(a, b, global_work_size: [n/(vector_length*block_size_x)], local_work_size: [local_work_size], repeat: repeat )
  }
  results.sort! { |r1, r2| r1[:duration] <=> r2[:duration] }
  res = results.first

  puts "Bandwidth = #{n*real_size*2*repeat/(res[:duration]*1e9)} GB/s"
  res[:duration]
}

puts config
puts "Bandwidth = #{n*real_size*2*repeat/(time*1e9)} GB/s"

