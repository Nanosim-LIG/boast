BOAST
=====

This section will present some simple examples to familiarize the user
with BOAST. More samples can be found in the git repository.

Installation
------------

BOAST is ruby based, so ruby needs to be installed on the machine.
Installation of boast can be done using the ruby built-in package
manager: *gem*. See following Listing for reference.

    ```bash
    sudo apt-get install ruby ruby-dev
    gem install --user-install BOAST
    ```

Variable and Procedure Declaration
----------------------------------

The following samples are presented using *irb* ruby interactive interpreter.
It can be launched using the *irb* command in a terminal.  Following
Listing shows the declaration of two variables of different kind.

    irb(main):001:0> require 'BOAST'
    => true
    irb(main):002:0> a = BOAST::Int "a"
    => a
    irb(main):003:0> b = BOAST::Real "b"
    => b
    irb(main):004:0> BOAST::decl a, b
    integer(kind=4) :: a
    real(kind=8) :: b
    => [a, b]

Following Listing shows the declaration of a procedure using the two previous
variables as parameters. For clarity irb echoes have been suppressed.

    005:0> p = BOAST::Procedure( "test_proc", [a,b] )
    006:0> BOAST::opn p
    SUBROUTINE test_proc(a, b)
      integer, parameter :: wp=kind(1.0d0)
      integer(kind=4) :: a
      real(kind=8) :: b
    007:0> BOAST::close p
    END SUBROUTINE test_proc

Switching Language
------------------

Following Listing shows how to switch BOAST to C.  Available languages are
*FORTRAN*, *C*, *CUDA* and *CL*.

    008:0> BOAST::lang = BOAST::C
    009:0> BOAST::opn p
    void test_proc(int32_t a, double b){
    010:0> BOAST::close p
    }

Defining a Complete Procedure
-----------------------------

Following Listing shows how to define a procedure and the associated code. Note
that here the parameters of the procedure have been associated a direction:
one, *a*, is an input parameter while the other, *b*, is an output parameter.

    011:0> BOAST::lang = BOAST::FORTRAN
    012:0> a = BOAST::Real( "a", :dir => :in)
    013:0> b = BOAST::Real( "b", :dir => :out)
    014:0> p = BOAST::Procedure( "plus_two", [a,b] ) {
    015:1*   BOAST::pr b === a + 2
    016:1> }
    017:0> BOAST::pr p
    SUBROUTINE plus_two(a, b)
      integer, parameter :: wp=kind(1.0d0)
      real(kind=8), intent(in) :: a
      real(kind=8), intent(out) :: b
      b = a + 2
    END SUBROUTINE plus_two
    018:0> BOAST::lang = BOAST::C
    019:0> BOAST::pr p
    void plus_two(const double a, double * b){
      (*b) = a + 2;
    }

Creating, Building and Running a Computing Kernel
-------------------------------------------------

Following Listing shows how to create a Computing kernel (*CKernel*) and build
it. Once a computing kernel is instantiated the output of BOAST will be
redirected to the computing kernel source code.  Line 4 sets the entry point of
the computing kernel to the procedure we just defined. By default compilation
commands are not shown unless an error occurs. This behavior can be changed by
switching to verbose mode.

When running the kernel all the arguments have to be specified. Running
a kernel returns a hash table containing information about the procedure
execution. In this simple case two informations are returned, first the
value of the output parameter *b* and second the time the kernel
execution took.

    020:0> BOAST::lang = BOAST::FORTRAN
    021:0> k = BOAST::CKernel::new
    022:0> BOAST::pr p
    023:0> k.procedure = p
    024:0> puts k
    SUBROUTINE plus_two(a, b)
      integer, parameter :: wp=kind(1.0d0)
      real(kind=8), intent(in) :: a
      real(kind=8), intent(out) :: b
      b = a + 2
    END SUBROUTINE plus_two
    025:0> k.build
    026:0> BOAST::verbose = true
    027:0> k.build
    gcc -O2 -Wall -fPIC -I/usr/lib/x86_64-linux-gnu/ruby/2.1.0 -I/usr/include/ruby-2.1.0 -I/usr/include/ruby-2.1.0/x86_64-linux-gnu -I/usr/include/x86_64-linux-gnu/ruby-2.1.0 -I/var/lib/gems/2.1.0/gems/narray-0.6.1.1 -DHAVE_NARRAY_H -c -o /tmp/Mod_plus_two20150309_4611_5a129k.o /tmp/Mod_plus_two20150309_4611_5a129k.c
    gfortran -O2 -Wall -fPIC -c -o /tmp/plus_two20150309-4611-5a129k.o /tmp/plus_two20150309-4611-5a129k.f90
    gcc -shared -o /tmp/Mod_plus_two20150309_4611_5a129k.so /tmp/Mod_plus_two20150309_4611_5a129k.o /tmp/plus_two20150309-4611-5a129k.o -Wl,-Bsymbolic-functions -Wl,-z,relro -rdynamic -Wl,-export-dynamic -L/usr/lib -lruby-2.1 -lrt
    028:0> r = k.run(5,0)
    029:0> puts r
    {:reference_return=>{:b=>7.0}, :duration=>5.84e-07}

Using Arrays in Procedures
--------------------------

Most computing kernels don't work on scalar values but rather on arrays
of data. Following Listing shows how to use arrays in computing
kernels. In this case we place ourselves in BOAST namespace to reduce
the syntax overhead. Variables *a* and *b* are one-dimensional arrays of
size *n*. Arrays in BOAST start at index 1 unless specified otherwise.
For instance `Dim(0,n-1)` would have created a dimension starting at 0.
Array bounds can also be negative and several dimensions can be
specified to obtain muti-dimensional arrays. For self contained
procedures/kernels one can use the shortcut written on line 13 to create
a CKernel object. As we are not specifying build options the build
command can also be omitted and will be automatically called when
running the kernel the first time. Lines 17 to 19 are used to check the
result of the kernel.

    001:0> require 'BOAST'
    002:0> require 'narray'
    003:0> include BOAST
    004:0> n = Int(  "n", :dir => :in )
    005:0> a = Real( "a", :dir => :in,  :dim => [Dim(n)] )
    006:0> b = Real( "b", :dir => :out, :dim => [Dim(n)] )
    007:0> p = Procedure( "plus_two", [n, a, b] ) {
    008:1*   decl i = Int( "i" )
    009:1>   pr For( i, 1, n ) {
    010:2*     pr b[i] === a[i] + 2.0
    011:2>   }
    012:1> }
    013:0> k = p.ckernel
    014:0> input  = NArray.float(1024).random
    015:0> output = NArray.float(1024)
    016:0> k.run(input.length, input, output)
    017:0> (output - input).each { |val|
    018:1*   raise "Error!" if (val-2).abs > 1e-15
    019:1> }
    020:0> stats = k.run(input.length, input, output)
    021:0> puts "Success, duration: #{stats[:duration]} s"
    Success, duration: 3.79e-06 s

The Canonical Case: Vector Addition
-----------------------------------

Following Listing shows the addition of two vectors in a third one. Here BOAST
is configured to have arrays starting at 0 and to use single precision reals by
default (Lines 5 and 6). The kernel declaration is encapsulated inside a method
to avoid cluttering the global namespace. Line 15 the expression `c[i] === a[i]
+ b[i]` is stored inside a variable *expr* for later use. Lines 16 to 23 show
that the kernel differs depending on the target language, in CUDA and OpenCL
each thread will process one element.

    ```ruby
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
    ```

Following Listing shows the a way to check the validity of the previous kernel
over the available range of languages. The options that are passed to run are
only relevant for GPU languages and are thus ignored in FORTRAN and C
(Line 16). Success is only printed if results are validated, else an exception
is raised (Lines 17 to 20).

    ```ruby
    n = 1024*1024
    a = NArray.sfloat(n).random
    b = NArray.sfloat(n).random
    c = NArray.sfloat(n)

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
    ```

