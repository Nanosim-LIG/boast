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
      pop_env( :default_real_size, :lang, :model )
    end
  end

  def test_add_int_real
    begin
      push_env( :default_real_size => 4, :lang => C, :model => :nehalem )
      a = Real :a, :vector_length => 4
      b = Int  :b, :vector_length => 4
      block = lambda { pr a + b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_ps( a, _mm_cvtepi32_ps( b ) );
EOF
      begin
        push_env( :architecture => ARM )
        assert_subprocess_output( <<EOF, "", &block )
vaddq_f32( a, vcvtq_s32_f32( b ) );
EOF
      ensure
        pop_env( :architecture )
      end
    ensure
      pop_env( :default_real_size, :lang, :model )
    end
  end

  def test_mul
    begin
      push_env( :default_real_size => 4, :lang => C, :model => :nehalem )
      a = Real :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      block = lambda { pr a * b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_mul_ps( a, b );
EOF
      begin
        push_env( :architecture => ARM )
        assert_subprocess_output( <<EOF, "", &block )
vmulq_f32( a, b );
EOF
      ensure
        pop_env( :architecture )
      end
    ensure
      pop_env( :default_real_size, :lang, :model )
    end
  end

  def test_mul_int_real
    begin
      push_env( :default_real_size => 4, :lang => C, :model => :nehalem )
      a = Int  :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      block = lambda { pr a * b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_mul_ps( _mm_cvtepi32_ps( a ), b );
EOF
      begin
        push_env( :architecture => ARM )
        assert_subprocess_output( <<EOF, "", &block )
vmulq_f32( vcvtq_s32_f32( a ), b );
EOF
      ensure
        pop_env( :architecture )
      end
    ensure
      pop_env( :default_real_size, :lang )
    end
  end

  def test_fma
    begin
      push_env( :default_real_size => 4, :lang => C, :model => :nehalem )
      a = Real :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      c = Real :c, :vector_length => 4
      block = lambda { pr FMA(a,b,c) }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_ps( c, _mm_mul_ps( a, b ) );
EOF
      begin
        push_env( :model => :haswell )
        assert_subprocess_output( <<EOF, "", &block )
_mm_fmadd_ps( a, b, c );
EOF
      ensure
        pop_env( :model )
      end
      begin
        push_env( :architecture => ARM )
        assert_subprocess_output( <<EOF, "", &block )
vmlaq_f32( c, a, b );
EOF
      ensure
        pop_env( :architecture )
      end
    ensure
      pop_env( :default_real_size, :lang, :model )
    end
  end

  def test_fms
    begin
      push_env( :default_real_size => 4, :lang => C, :model => :nehalem )
      a = Real :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      c = Real :c, :vector_length => 4
      block = lambda { pr FMS(a,b,c) }
      assert_subprocess_output( <<EOF, "", &block )
_mm_sub_ps( c, _mm_mul_ps( a, b ) );
EOF
      begin
        push_env( :model => :haswell )
        assert_subprocess_output( <<EOF, "", &block )
_mm_fnmadd_ps( a, b, c );
EOF
      ensure
        pop_env( :model )
      end
      begin
        push_env( :architecture => ARM )
        assert_subprocess_output( <<EOF, "", &block )
vmlsq_f32( c, a, b );
EOF
      ensure
        pop_env( :architecture )
      end
    ensure
      pop_env( :default_real_size, :lang, :model )
    end
  end

end
