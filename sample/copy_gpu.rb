require 'BOAST'
require 'narray_ffi'

include BOAST
BOAST.set_array_start(0)


def kernel_transpose
  
  d1 = Int :d1, signed: false
  d2 = Int :d2, signed: false
  c = Int :c, dir: :out, dim: [ Dim(d1), Dim(d2) ]

  i = Int :i, signed: false
  j = Int :j, signed: false
  p = Procedure(  "init_kern", [d1, d2, c] ) {
    decl i, j
    pr i === get_global_id(0)
    pr j === get_global_id(1)
    pr c[i, j] === i * d2 + j
  }
  p.ckernel
end



def kernel_init
  
  c = Int :c, dir: :out, dim: [ Dim() ]

  i = Int :i, signed: false
  p = Procedure(  "init_kern", [c] ) {
    decl i
    pr i === get_global_id(0)
    pr c[i] === i 
  }
  p.ckernel
end

def kernel_copy(indirect = false)
  function_name = "copy_kern"
  n = Int :n, signed: false
  a = Int :a, dir: :in, dim: [ Dim(0, n-1) ]
  b = Int :b, dir: :out, dim: [ Dim(0, n-1) ]
  c = Int :c, dir: :in, dim: [ Dim(0, n-1) ]

  i = Int :i, signed: false

  args = [n, a, b]
  args.push c if indirect

  p = Procedure( function_name, args ) {
    decl i
    pr i === get_global_id(0)
    pr b[i] === (indirect ? a[c[i]] : a[i])
  }

  p.ckernel
end

BOAST::set_lang( BOAST::CUDA )

n = 1<<28
slice = 1<<14 

a = NArray.sfloat(n).random!
b = NArray.sfloat(n).random!
c = NArray.int(n)

k_init = kernel_init
puts k_init
k_init.run(c, :global_work_size => [n], :local_work_size => [128] )

p c[-15..-1]

p a

k = kernel_copy
puts k

k.build

res = k.run(n, a, b, :global_work_size => [n], :local_work_size => [128] )
res = k.run(n, a, b, :global_work_size => [n], :local_work_size => [128] )

p res

puts "Bandwidth = #{n*4*2/(res[:duration]*1e9)} GB/s"

err = b - a
puts "Error: #{err.abs.max}"

b.random!

k = kernel_copy(true)

puts k

k.build

res = k.run(n, a, b, c, :global_work_size => [n], :local_work_size => [128] )
res = k.run(n, a, b, c, :global_work_size => [n], :local_work_size => [128] )

p res

puts "Bandwidth = #{n*4*3/(res[:duration]*1e9)} GB/s"

err = b - a
puts "Error: #{err.abs.max}"

k_init = kernel_transpose
puts k_init
k_init.run(slice, slice, c, :global_work_size => [slice, slice], :local_work_size => [16,16] )

k = kernel_copy(true)

puts k

k.build

res = k.run(n, a, b, c, :global_work_size => [n], :local_work_size => [128] )
res = k.run(n, a, b, c, :global_work_size => [n], :local_work_size => [128] )

p res

puts "Bandwidth = #{n*4*3/(res[:duration]*1e9)} GB/s"

err = b.reshape(slice, slice) - a.reshape(slice, slice).transpose(1,0)
puts "Error: #{err.abs.max}"


