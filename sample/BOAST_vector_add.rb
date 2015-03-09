require 'narray'
require 'BOAST'
include BOAST
set_array_start( 0 )
def vector_add
  kernel = CKernel::new
  n = Int("n",{:dir => :in, :signed => false})
  a = Real("a",{:dir => :in, :dim => [ Dim(n)] })
  b = Real("b",{:dir => :in, :dim => [ Dim(n)] })
  c = Real("c",{:dir => :out, :dim => [ Dim(n)] })
  i = Int( :i, :signed => false )
  ig = Sizet :ig
  if get_lang == CL then
    @@output.puts "#pragma OPENCL EXTENSION cl_khr_fp64: enable"
  end
  pr p = Procedure("vector_add", [n,a,b,c]) {
    if (get_lang == CL or get_lang == CUDA) then
      decl ig
      pr ig === get_global_id(0)
      pr c[ig] === a[ig] + b[ig]
    else
      decl i
      pr For(i,0,n-1) {
        pr c[i] === a[i] + b[i]
      }
    end
  }
  kernel.procedure = p
  return kernel
end

n = 1024*1024
a = NArray.float(n).random
b = NArray.float(n).random
c = NArray.float(n)
c_ref = NArray.float(n)

epsilon = 10e-15

c_ref = a + b

[:FORTRAN, :C, :CL, :CUDA].each { |l|
  set_lang( BOAST.const_get(l)  )
  puts l
  k = vector_add
  puts k.print
  c.random!
  if lang == CL or lang == CUDA then
    k.run(n, a, b, c, :global_work_size => [n,1,1], :local_work_size => [32,1,1])
  else
    k.run(n, a, b, c)
  end
  diff = (c_ref - c).abs
  diff.each { |elem|
    raise "Warning: residue too big: #{elem}" if elem > epsilon
  }
}
