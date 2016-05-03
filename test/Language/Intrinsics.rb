require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestIntrinsics < Minitest::Test

  def test_coverage
    cov = Intrinsics.check_coverage
    puts cov if cov.size > 0
    assert_equal(0, cov.size)
  end

  def test_add
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      block = lambda { pr a + b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_ps( a, b );
EOF
      push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
vaddq_f32( a, b );
EOF
      }
    }
  end

  def test_add_knl
    push_env( :default_real_size => 8, :lang => C, :model => :knl, :architecture => X86 ) {
      a = Real :a, :vector_length => 8
      b = Real :b, :vector_length => 8
      block = lambda { pr a + b }
      assert_subprocess_output( <<EOF, "", &block )
_mm512_add_pd( a, b );
EOF
      push_env( :architecture => ARM ) {
        assert_raises( IntrinsicsError, &block )
      }
    }
  end

  def test_add_int_real
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 4
      b = Int  :b, :vector_length => 4, :size => 2
      block = lambda { pr a + b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_ps( a, _mm_cvtepi32_ps( _mm_cvtepi16_epi32( b ) ) );
EOF
      push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
vaddq_f32( a, vcvtq_f32_s32( vmovl_s16( b ) ) );
EOF
      }
    }
  end

  def test_add_int_real_knl
    push_env( :default_real_size => 8, :lang => C, :model => :knl, :architecture => X86 ) {
      a = Real :a, :vector_length => 8
      b = Int  :b, :vector_length => 8, :size => 2
      block = lambda { pr a + b }
      assert_subprocess_output( <<EOF, "", &block )
_mm512_add_pd( a, _mm512_cvtepi32_pd( _mm256_cvtepi16_epi32( b ) ) );
EOF
      push_env( :architecture => ARM ) {
        assert_raises( IntrinsicsError, &block )
      }
    }
  end

  def test_mul
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      block = lambda { pr a * b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_mul_ps( a, b );
EOF
      push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
vmulq_f32( a, b );
EOF
      }
    }
  end

  def test_mul_int_real
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Int  :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      block = lambda { pr a * b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_mul_ps( _mm_cvtepi32_ps( a ), b );
EOF
      push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
vmulq_f32( vcvtq_f32_s32( a ), b );
EOF
      }
    }
  end

  def test_fma
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      c = Real :c, :vector_length => 4
      block = lambda { pr FMA(a,b,c) }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_ps( c, _mm_mul_ps( a, b ) );
EOF
      push_env( :model => :haswell ) {
        assert_subprocess_output( <<EOF, "", &block )
_mm_fmadd_ps( a, b, c );
EOF
      }
      push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
vmlaq_f32( c, a, b );
EOF
      }
    }
  end

  def test_fms
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      c = Real :c, :vector_length => 4
      block = lambda { pr FMS(a,b,c) }
      assert_subprocess_output( <<EOF, "", &block )
_mm_sub_ps( c, _mm_mul_ps( a, b ) );
EOF
      push_env( :model => :haswell ) {
        assert_subprocess_output( <<EOF, "", &block )
_mm_fnmadd_ps( a, b, c );
EOF
      }
      push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
vmlsq_f32( c, a, b );
EOF
      }
    }
  end

  def test_load
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block1 = lambda { pr b === a[0] }
      block2 = lambda { pr b === Load(a[0],b) }
      [block1, block2].each { |block|
        assert_subprocess_output( <<EOF, "", &block )
b = _mm_loadu_ps( &a[0] );
EOF
        push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
b = vldlq_f32( &a[0] );
EOF
        }
      }
    }
  end

  def test_load_aligned
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      a_index = a[0].set_align(16)
      block1 = lambda { pr b === a_index }
      block2 = lambda { pr b === Load(a_index,b) }
      [block1, block2].each { |block|
        assert_subprocess_output( <<EOF, "", &block )
b = _mm_load_ps( &a[0] );
EOF
        push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
b = vldlq_f32( &a[0] );
EOF
        }
      }
    }
  end

  def test_mask_load
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :sandybridge, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block = lambda { pr b === MaskLoad(a[0], [1, 0, 1, 0], b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_maskload_ps( (float * ) &a[0], _mm_setr_epi32( -1, 0, -1, 0 ) );
EOF
    }
  end

  def test_zeromask_load
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :sandybridge, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block = lambda { pr b === MaskLoad(a[0], [0, 0, 0, 0], b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_setzero_ps( );
EOF
    }
  end

  def test_fullmask_load
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :sandybridge, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block = lambda { pr b === MaskLoad(a[0], [1, 1, 1, 1], b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_loadu_ps( &a[0] );
EOF
    }
  end

  def test_store
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block1 = lambda { pr a[0] === b }
      block2 = lambda { pr Store(a[0], b) }
      [block1, block2].each { |block|
        assert_subprocess_output( <<EOF, "", &block )
_mm_storeu_ps( (float * ) &a[0], b );
EOF
        push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
vstlq_f32( (float * ) &a[0], b );
EOF
        }
      }
    }
  end

  def test_store_expression
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block1 = lambda { pr a[0] === b + b }
      block2 = lambda { pr Store(a[0], b + b) }
      [block1, block2].each { |block|
        assert_subprocess_output( <<EOF, "", &block )
_mm_storeu_ps( (float * ) &a[0], _mm_add_ps( b, b ) );
EOF
        push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
vstlq_f32( (float * ) &a[0], vaddq_f32( b, b ) );
EOF
        }
      }
    }
  end

  def test_store_aligned
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      a_index = a[0].set_align(16)
      block1 = lambda { pr a_index === b }
      block2 = lambda { pr Store(a_index, b) }
      [block1, block2].each { |block|
        assert_subprocess_output( <<EOF, "", &block )
_mm_store_ps( (float * ) &a[0], b );
EOF
        push_env( :architecture => ARM ) {
        assert_subprocess_output( <<EOF, "", &block )
vstlq_f32( (float * ) &a[0], b );
EOF
        }
      }
    }
  end

  def test_mask_store
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :sandybridge, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block = lambda { pr MaskStore(a[0], b, [1, 0, 1, 0]) }
      assert_subprocess_output( <<EOF, "", &block )
_mm_maskstore_ps( (float * ) &a[0], _mm_setr_epi32( -1, 0, -1, 0 ), b );
EOF
    }
  end

  def test_fullmask_store
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :sandybridge, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block = lambda { pr MaskStore(a[0], b, [1, 1, 1, 1]) }
      assert_subprocess_output( <<EOF, "", &block )
_mm_storeu_ps( (float * ) &a[0], b );
EOF
    }
  end

  def test_zeromask_store
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :sandybridge, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block = lambda { pr MaskStore(a[0], b, [0, 0, 0, 0]) }
      assert_subprocess_output( "", "", &block )
    }
  end

  def test_set1
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      b = Real :b, :vector_length => 4
      block = lambda { pr b === Set(1.0, b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_set1_ps( 1.0 );
EOF
      push_env( :architecture => ARM ) {
      assert_subprocess_output( <<EOF, "", &block )
b = vdupq_n_f32( 1.0 );
EOF
      }
    }
  end

  def test_set1_0
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      b = Real :b, :vector_length => 4
      block = lambda { pr b === Set(0.0, b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_setzero_ps( );
EOF
      push_env( :architecture => ARM ) {
      assert_subprocess_output( <<EOF, "", &block )
b = vdupq_n_f32( 0.0 );
EOF
      }
    }
  end

  def test_set
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      b = Real :b, :vector_length => 4
      block = lambda { pr b === Set([1.0, 2.0, 3.0, 4.0], b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_setr_ps( 1.0, 2.0, 3.0, 4.0 );
EOF
      push_env( :architecture => ARM ) {
      assert_subprocess_output( <<EOF, "", &block )
b = vsetq_lane_f32( 4.0, vsetq_lane_f32( 3.0, vsetq_lane_f32( 2.0, vsetq_lane_f32( 1.0, vdupq_n_f32( 0 ), 0 ), 1 ), 2 ), 3 );
EOF
      }
    }
  end

  def test_set_identical
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      b = Real :b, :vector_length => 4
      block = lambda { pr b === Set([1.0, 1.0, 1.0, 1.0], b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_set1_ps( 1.0 );
EOF
      push_env( :architecture => ARM ) {
      assert_subprocess_output( <<EOF, "", &block )
b = vdupq_n_f32( 1.0 );
EOF
      }
    }
  end

  def test_set_zero_array
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      b = Real :b, :vector_length => 4
      block = lambda { pr b === Set([0.0, 0.0, 0.0, 0.0], b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_setzero_ps( );
EOF
      push_env( :architecture => ARM ) {
      assert_subprocess_output( <<EOF, "", &block )
b = vdupq_n_f32( 0.0 );
EOF
      }
    }
  end

end
