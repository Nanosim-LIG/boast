require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class Vectors < Minitest::Test

  def test_decl_vector
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 4
      block = lambda { decl a }
      assert_subprocess_output( <<EOF, "", &block )
__m128 a;
EOF
      push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
float32x4_t a;
EOF
      }
      push_env( :lang => FORTRAN ) {
        assert_subprocess_output( <<EOF, "", &block )
real(kind=4), dimension(4) :: a
!DIR$ ATTRIBUTES ALIGN: 16:: a
EOF
      }
    }
  end

  def test_vector_array
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      n = Int :n
      a = Real :a, :vector_length => 4, :dim => Dim( n )
      b = Real :b, :vector_length => 4
      block = lambda { 
        decl a, b
        pr b === a[3] + a[4]
        pr a[3] === b
      }
      assert_subprocess_output( <<EOF, "", &block )
__m128 * a;
__m128 b;
b = _mm_add_ps( a[3 - (1)], a[4 - (1)] );
a[3 - (1)] = b;
EOF
      push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
float32x4_t * a;
float32x4_t b;
b = vaddq_f32( a[3 - (1)], a[4 - (1)] );
a[3 - (1)] = b;
EOF
      }
      push_env( :lang => FORTRAN ) {
        assert_subprocess_output( <<EOF, "", &block )
real(kind=4), dimension(4, n) :: a
!DIR$ ATTRIBUTES ALIGN: 16:: a
real(kind=4), dimension(4) :: b
!DIR$ ATTRIBUTES ALIGN: 16:: b
b = a(:, 3) + a(:, 4)
a(:, 3) = b
EOF
      }
    }
  end

  def test_load_vector
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      n = Int :n
      a = Real :a, :dim => Dim( n )
      b = Real :b, :vector_length => 4
      block = lambda { pr b === Load(a[3],b); pr b === a[3]; pr b === a[n-3] }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_loadu_ps( &a[3 - (1)] );
b = _mm_loadu_ps( &a[3 - (1)] );
b = _mm_loadu_ps( &a[n - (3) - (1)] );
EOF
      push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
b = vldlq_f32( &a[3 - (1)] );
b = vldlq_f32( &a[3 - (1)] );
b = vldlq_f32( &a[n - (3) - (1)] );
EOF
      }
      push_env( :lang => FORTRAN ) {
        assert_subprocess_output( <<EOF, "", &block )
b = a(3:6)
b = a(3:6)
b = a(n - (3):n - (3) + 4 - (1))
EOF
      }
    }
  end

  def test_store_vector
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      n = Int :n
      a = Real :a, :dim => Dim( n )
      b = Real :b, :vector_length => 4
      block = lambda { pr Store(a[3], b); pr a[3] === b; pr a[n-3] === b; }
      assert_subprocess_output( <<EOF, "", &block )
_mm_storeu_ps( (float * ) &a[3 - (1)], b );
_mm_storeu_ps( (float * ) &a[3 - (1)], b );
_mm_storeu_ps( (float * ) &a[n - (3) - (1)], b );
EOF
      push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
vstlq_f32( (float * ) &a[3 - (1)], b );
vstlq_f32( (float * ) &a[3 - (1)], b );
vstlq_f32( (float * ) &a[n - (3) - (1)], b );
EOF
      }
      push_env( :lang => FORTRAN ) {
        assert_subprocess_output( <<EOF, "", &block )
a(3:6) = b
a(3:6) = b
a(n - (3):n - (3) + 4 - (1)) = b
EOF
      }
    }
  end

end
