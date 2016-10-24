require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class Vectors < Minitest::Test

  def test_slice
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86, :use_vla => true ) {
      a = Real :a, :dim => Dim()
      block = lambda { pr a.slice(2..5) === 5 }
      assert_subprocess_output( <<EOF, "", &block )
a[2 - (1):5 - (2) + 1] = 5;
EOF
      push_env( :lang => FORTRAN ) {
        assert_subprocess_output( <<EOF, "", &block )
a(2:5) = 5
EOF
      }
    }
  end

end
