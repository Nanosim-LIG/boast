[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'narray_ffi'
require 'BOAST'
include BOAST


i = Int :i
j = Int :j

a_min = 1
a_max = 100
b_min = 1
b_max = 10

x = Real( :x, :dim => [Dim(a_min, a_max), Dim(b_min, b_max)] )
y = Real( :y, :dim => [Dim(a_min, a_max), Dim(b_min, b_max)] )

for_j = nil
body = nil

for_i = For(i, a_min, a_max) {
  for_j = For(j, b_min, b_max) {
    body = lambda {
      pr y[i,j] === x[i,j]
    }
    body.call
  }
  pr for_j
}
pr for_i

def permute(loop1, loop2)
  loop1.block = loop2.block
  loop2.block = lambda {
    pr loop1
  }
  return loop2
end
pr permute(for_i, for_j)

def permutations(body, *loops)
  permutations = loops.permutation

  permutations.each { |loops|
    loops.last.block = body
    loops.each_cons(2) { |l1, l2|
      l1.block = lambda {
        pr l2
      }
    }
    pr loops.first
  }
end

permutations(body, for_i, for_j)
