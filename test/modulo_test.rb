[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'BOAST'
include BOAST

def modulo_kernel(options = {})
  set_lang(options[:lang])
  a = Int :a, :dir => :in
  b = Int :b, :dir => :in
  c = Int :c
  p = Procedure( :test_mod, [a,b], :return => c) {
    pr c === Modulo(a, b)
  }
  return p.ckernel
end

def test_modulo(a, b)
  puts "a: #{a} b: #{b}"
  puts "Ruby:    #{a%b}"
  puts "FORTRAN: #{$kern_f.run(a,b)[:return]}"
  puts "C:       #{$kern_c.run(a,b)[:return]}"
end

$kern_c = modulo_kernel( lang: C )
puts $kern_c
$kern_f = modulo_kernel( lang: FORTRAN )
puts $kern_f

test_modulo( 16, 5 )
test_modulo( -16, 5 )
test_modulo( 16, -5 )
test_modulo( -16, -5 )

def modulo_kernel(options = {})
  set_lang(options[:lang])
  a = Real :a, :dir => :in
  b = Real :b, :dir => :in
  c = Real :c
  p = Procedure( :test_mod, [a,b], :return => c, :include => "math.h") {
    pr c === Modulo(a, b)
  }
  return p.ckernel
end

$kern_c = modulo_kernel( lang: C )
puts $kern_c
$kern_f = modulo_kernel( lang: FORTRAN )
puts $kern_f

test_modulo( 16.2, 5.3 )
test_modulo( -16.2, 5.3 )
test_modulo( 16.2, -5.3 )
test_modulo( -16.2, -5.3 )

set_default_real_size(4)

$kern_c = modulo_kernel( lang: C )
puts $kern_c
$kern_f = modulo_kernel( lang: FORTRAN )
puts $kern_f

test_modulo( 16.2, 5.3 )
test_modulo( -16.2, 5.3 )
test_modulo( 16.2, -5.3 )
test_modulo( -16.2, -5.3 )

def modulo_kernel(options = {})
  set_lang(options[:lang])
  a = Real :a, :dir => :in, :size => 8
  b = Real :b, :dir => :in
  c = Real :c
  p = Procedure( :test_mod, [a,b], :return => c, :include => "math.h") {
    pr c === Modulo(a, b)
  }
  return p.ckernel
end

$kern_c = modulo_kernel( lang: C )
puts $kern_c
$kern_f = modulo_kernel( lang: FORTRAN )
puts $kern_f

test_modulo( 16.2, 5.3 )
test_modulo( -16.2, 5.3 )
test_modulo( 16.2, -5.3 )
test_modulo( -16.2, -5.3 )

def modulo_kernel(options = {})
  set_lang(options[:lang])
  a = Real :a, :dir => :in, :size => 8
  b = Int :b, :dir => :in
  c = Real :c
  p = Procedure( :test_mod, [a,b], :return => c, :include => "math.h") {
    pr c === Modulo(a, b)
  }
  return p.ckernel
end

$kern_c = modulo_kernel( lang: C )
puts $kern_c
$kern_f = modulo_kernel( lang: FORTRAN )
puts $kern_f

test_modulo( 16.2, 5 )
test_modulo( -16.2, 5 )
test_modulo( 16.2, -5 )
test_modulo( -16.2, -5 )

def modulo_kernel(options = {})
  set_lang(options[:lang])
  a = Int :a, :dir => :in, :size => 8
  b = Real :b, :dir => :in
  c = Real :c
  p = Procedure( :test_mod, [a,b], :return => c, :include => "math.h") {
    pr c === Modulo(a, b)
  }
  return p.ckernel
end

$kern_c = modulo_kernel( lang: C )
puts $kern_c
$kern_f = modulo_kernel( lang: FORTRAN )
puts $kern_f

test_modulo( 16, 5.3 )
test_modulo( -16, 5.3 )
test_modulo( 16, -5.3 )
test_modulo( -16, -5.3 )


