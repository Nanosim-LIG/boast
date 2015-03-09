require 'narray'
require 'BOAST'
include BOAST

set_array_start(0)
set_default_real_size(4)

def vector_add
  n = Int("n",:dir => :in)
  a = Real("a",:dir => :in, :dim => [ Dim(n)] )
  b = Real("b",:dir => :in, :dim => [ Dim(n)] )
  c = Real("c",:dir => :out, :dim => [ Dim(n)] )
  p = Procedure("vector_add", [n,a,b,c]) {
    decl i = Int("i")
    expr = c[i] === a[i] + b[i]
    if (get_lang == CL or get_lang == CUDA) then
      pr i === get_global_id(0)
      pr expr
    else
      pr For(i,0,n-1) {
        pr expr
      }
    end
  }
  return p.ckernel
end

n = 1024*1024
a = NArray.sfloat(n).random
b = NArray.sfloat(n).random
c = NArray.sfloat(n)
c_ref = NArray.float(n)

epsilon = 10e-15

c_ref = a + b

[:FORTRAN, :C, :CL, :CUDA].each { |l|
  set_lang( BOAST.const_get(l)  )
  puts "#{l}:"
  k = vector_add
  puts k.print
  c.random!
  k.run(n, a, b, c, :global_work_size => [n,1,1], :local_work_size => [32,1,1])
  diff = (c_ref - c).abs
  diff.each { |elem|
    raise "Warning: residue too big: #{elem}" if elem > epsilon
  }
}
puts "Success!"
