require 'rubygems'
require 'narray'
require 'BOAST'
module BOAST
  def BOAST::vector_add
    kernel = CKernel::new
    function_name = "vector_add"
    n = Int("n",{:dir => :in, :signed => false})
    a = Real("a",{:dir => :in, :dim => [ Dim(0,n-1)] })
    b = Real("b",{:dir => :in, :dim => [ Dim(0,n-1)] })
    c = Real("c",{:dir => :out, :dim => [ Dim(0,n-1)] })
    i = Int :i, :signed => false 
    ig = Sizet :ig
    rts = CStruct("rts",:type_name => "recursive_test_struct", :members => [Int("ig"),i])
    ts = CStruct("ts", :type_name => "test_struct", :members => [Real("r"),rts])
    if get_lang == CL then
      @@output.puts "#pragma OPENCL EXTENSION cl_khr_fp64: enable"
    end
    if get_lang != FORTRAN then
      rts.type.define
      ts.type.define
    end
    print p = Procedure(function_name, [n,a,b,c]) {
      if get_lang == FORTRAN then
        rts.type.define
        ts.type.define
      end
      decl ts
      if (get_lang == CL or get_lang == CUDA) then
        decl ig
        print ig === get_global_id(0)
        print c[ig] === a[ig] + b[ig]
      else
        decl i
        print For(i,0,n-1) {
          print c[i] === a[i] + b[i]
        }
      end
      print ts.r === 0.0
      print ts.rts.i === 0
    }
    kernel.procedure = p
    return kernel
  end
end

def rndup( val, div)
  return (val%div) == 0 ? val : val + div - (val%div)
end


n = 1024*1024
a = NArray.float(n).random
b = NArray.float(n).random
c = NArray.float(n)
c_ref = NArray.float(n)

epsilon = 10e-15

BOAST::set_lang( BOAST::FORTRAN )
puts "FORTRAN"
k = BOAST::vector_add
puts k.print
k.run(n,a,b,c_ref)
BOAST::set_lang( BOAST::C )
puts "C"
c.random!
k = BOAST::vector_add
puts k.print
k.run(n,a,b,c)
diff = (c_ref - c).abs
diff.each { |elem|
  raise "Warning: residue too big: #{elem}" if elem > epsilon
}
BOAST::set_lang( BOAST::CL )
puts "CL"
c.random!
k = BOAST::vector_add
puts k.print
k.run(n, a, b, c, :global_work_size => [rndup(n,32), 1,1], :local_work_size => [32,1,1] )
diff = (c_ref - c).abs
diff.each { |elem|
  raise "Warning: residue too big: #{elem}" if elem > epsilon
}
BOAST::set_lang( BOAST::CUDA )
puts "CUDA"
c.random!
k = BOAST::vector_add
puts k.print
k.build(:LDFLAGS => " -L/usr/local/cuda-5.5.22/lib64")
k.run(n, a, b, c, :block_number => [rndup(n,32)/32, 1,1], :block_size => [32,1,1] )
diff = (c_ref - c).abs
diff.each { |elem|
  raise "Warning: residue too big: #{elem}" if elem > epsilon
}
