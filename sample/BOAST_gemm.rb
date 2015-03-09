require 'narray'
require 'BOAST'
module BOAST
  def BOAST::gemm(unroll=1)
    kernel = CKernel::new
    function_name = "gemm"
    n = Int("n",  :dir => :in)
    m = Int("m",  :dir => :in)
    l = Int("l",  :dir => :in)
    a = Real("a",  :dir => :in, :dim => [ Dim(l), Dim(n)] )
    b = Real("b",  :dir => :in, :dim => [ Dim(l), Dim(m)] )
    c = Real("c",  :dir => :out, :dim => [ Dim(n), Dim(m)] )
    print p = Procedure(function_name, [n,m,l,a,b,c]) {
      i = Int :i
      j = Int :j
      k = Int :k
      w = Int :w
      sum = Real :sum
      decl i,j,k,sum
      print For(i,1,n) {
        print For(j,1,m) {
          print sum === 0.0
          print For(k,1,l,:step => unroll) {
            block = lambda {
              print sum === sum + a[k + w,i] * b[k + w,j]
            }

            f = For(w, 0, unroll-1, &block)
            f.unroll
          }
          print c[i,j] === sum
        }
      }
    }
    kernel.procedure = p
    return kernel
  end
end

n = 1024
m = 1024
l = 1024

a = NArray.float(n,l).random
b = NArray.float(m,l).random
c = NArray.float(m,n)
c_ref = NArray.float(m,n)

epsilon = 10e-15

BOAST::set_lang( BOAST::FORTRAN )
puts "FORTRAN"
k = BOAST::gemm(8)
puts k.print
stats = []
3.times { stats.push k.run(n,m,l,a,b,c_ref)[:duration] }
puts stats
BOAST::set_lang( BOAST::C )
puts "C"
c.random!
k = BOAST::gemm(8)
puts k.print
stats = []
3.times { stats.push k.run(n,m,l,a,b,c)[:duration] }
puts stats
diff = (c_ref - c).abs
diff.each { |elem|
  raise "Warning: residue too big: #{elem}" if elem > epsilon
}
