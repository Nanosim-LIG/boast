require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class Vectors < Minitest::Test

  def test_slice
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86, :use_vla => true ) {
      a = Real :a, :dim => Dim()
      blocks = [ lambda { pr a.slice(2..5) === 5 }, lambda { pr a.slice([2,5]) === 5 } ]
      blocks.each { |block|
        assert_subprocess_output( <<EOF, "", &block )
a[2 - (1):5 - (2) + 1] = 5;
EOF
        push_env( :lang => FORTRAN ) {
          assert_subprocess_output( <<EOF, "", &block )
a(2:5) = 5
EOF
        }
      }
    }
  end

  def test_slice_multi_dim
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86, :use_vla => true ) {
      n = Int :n
      m = Int :m
      a = Real :a, :dim => [Dim(n), Dim(m), Dim()]
      blocks = [ lambda { pr a.slice(5,2..5,:all) === 5 }, lambda { pr a.slice(5,[2,5],nil) === 5 } ]
      blocks.each { |block|
        assert_subprocess_output( <<EOF, "", &block )
a[:][2 - (1):5 - (2) + 1][5 - (1)] = 5;
EOF
        push_env( :lang => FORTRAN ) {
          assert_subprocess_output( <<EOF, "", &block )
a(5,2:5,:) = 5
EOF
        }
      }
    }
  end

  def test_slice_exclusive_range
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86, :use_vla => true ) {
      a = Real :a, :dim => Dim()
      block = lambda { pr a.slice(2...5) === 5 }
      assert_subprocess_output( <<EOF, "", &block )
a[2 - (1):5 - (2)] = 5;
EOF
      push_env( :lang => FORTRAN ) {
        assert_subprocess_output( <<EOF, "", &block )
a(2:5 - (1)) = 5
EOF
      }
    }
  end

  def test_slice_step
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86, :use_vla => true ) {
      a = Real :a, :dim => Dim()
      block = lambda { pr a.slice([2,6,2]) === 5 }
      assert_subprocess_output( <<EOF, "", &block )
a[2 - (1):(6 - (2) + 1) / (2):2] = 5;
EOF
      push_env( :lang => FORTRAN ) {
        assert_subprocess_output( <<EOF, "", &block )
a(2:6:2) = 5
EOF
      }
    }
  end

end
