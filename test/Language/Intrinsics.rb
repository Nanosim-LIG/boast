require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestIntrinsics < Minitest::Test

  def test_add
    begin
      push_env( :default_real_size => 4, :lang => C, :model => :nehalem )
      a = Real :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      block = lambda { pr a + b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_ps( a, b );
EOF
      begin
        push_env( :architecture => ARM )
        assert_subprocess_output( <<EOF, "", &block )
vaddq_f32( a, b );
EOF
      ensure
        pop_env( :architecture )
      end
    ensure
      pop_env( :default_real_size, :lang )
    end
  end

end
