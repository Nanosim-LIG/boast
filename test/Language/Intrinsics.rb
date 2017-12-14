require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative '../helper'

class TestIntrinsics < Minitest::Test

  def test_coverage
    covs = Intrinsics.check_coverage
    covs.each { |cov|
      puts cov if cov.size > 0
      assert_equal(0, cov.size)
    }
  end

  def test_add
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      block = lambda { pr a + b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_ps( a, b );
EOF
      push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
vaddq_f32( a, b );
EOF
      }
    }
  end

  def test_add_double
    push_env( :default_real_size => 8, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 2
      b = Real :b, :vector_length => 2
      block = lambda { pr a + b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_pd( a, b );
EOF
      push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_raises( IntrinsicsError, &block )
      }
      push_env( :architecture => ARM, :model => "armv8-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
vaddq_f64( a, b );
EOF
      }
    }
  end

  def test_max
    push_env( :default_real_size => 4, :lang => FORTRAN, :model => :nehalem, :architecture => X86 ) {
      a = Real :a
      b = Real :b
      block = lambda { pr a === Max(a, b); pr a === Min(a, 1.0) }
      assert_subprocess_output( <<EOF, "", &block )
a = max( a, b )
a = min( a, 1.0 )
EOF
      [ C, CL, CUDA ].each { |l|
        set_lang( l )
        assert_subprocess_output( <<EOF, "", &block )
a = max( a, b );
a = min( a, 1.0f );
EOF
      }
    }
  end

  def test_max_vect
    push_env( :default_real_size => 4, :lang => FORTRAN, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 4
      b = Real :b, :vector_length => 4
      block = lambda { pr a === Max(a, b); pr a === Min(a, Set(1.0, a)); pr a === Min(a, 1.0) }
      assert_subprocess_output( <<EOF, "", &block )
a = max( a, b )
a = min( a, 1.0 )
a = min( a, 1.0 )
EOF
      set_lang( C )
      assert_subprocess_output( <<EOF, "", &block )
a = _mm_max_ps( a, b );
a = _mm_min_ps( a, _mm_set1_ps( 1.0f ) );
a = _mm_min_ps( a, _mm_set1_ps( 1.0f ) );
EOF
      set_lang( CL )
      assert_subprocess_output( <<EOF, "", &block )
a = max( a, b );
a = min( a, (float4)( 1.0f ) );
a = min( a, 1.0f );
EOF
      [ CUDA ].each { |l|
        set_lang( l )
        assert_subprocess_output( <<EOF, "", &block )
a = max( a, b );
a = min( a, 1.0f );
a = min( a, 1.0f );
EOF
      }
    }
  end

  def test_sqrt
    push_env( :default_real_size => 4, :lang => FORTRAN, :model => :nehalem, :architecture => X86 ) {
      a = Real :a
      b = Real :b, :vector_length => 4
      c = Real :c, :size => 8
      block = lambda { pr Sqrt(b); pr Sqrt(a); pr Sqrt(c) }
      assert_subprocess_output( <<EOF, "", &block )
sqrt( b )
sqrt( a )
sqrt( c )
EOF
      set_lang( CUDA )
      assert_subprocess_output( <<EOF, "", &block )
sqrtf( b );
sqrtf( a );
sqrt( c );
EOF
      set_lang( CL )
      assert_subprocess_output( <<EOF, "", &block )
sqrt( b );
sqrt( a );
sqrt( c );
EOF
      set_lang( C )
      assert_subprocess_output( <<EOF, "", &block )
_mm_sqrt_ps( b );
sqrtf( a );
sqrt( c );
EOF
      push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_raises( IntrinsicsError, "Vector square root unsupported on ARM architecture!", &block )
      }
    }
  end

  def test_sin
    push_env( :default_real_size => 4, :lang => FORTRAN, :model => :nehalem, :architecture => X86 ) {
      a = Real :a
      b = Real :b, :vector_length => 4
      c = Real :c, :size => 8
      block = lambda { pr Sin(b); pr Sin(a); pr Sin(c) }
      assert_subprocess_output( <<EOF, "", &block )
sin( b )
sin( a )
sin( c )
EOF
      set_lang( CUDA )
      assert_subprocess_output( <<EOF, "", &block )
sinf( b );
sinf( a );
sin( c );
EOF
      set_lang( CL )
      assert_subprocess_output( <<EOF, "", &block )
sin( b );
sin( a );
sin( c );
EOF
      set_lang( C )
      assert_subprocess_output( <<EOF, "", &block )
_mm_sin_ps( b );
sinf( a );
sin( c );
EOF
      push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_raises( IntrinsicsError, "Vector square root unsupported on ARM architecture!", &block )
      }
    }
  end

  def test_vec_sin
    push_env( :default_real_size => 4, :lang => C, :model => :haswell, :architecture => X86 ) {
      a = Real :a, :size => 8, :vector_length => 4
      b = Real :b, :vector_length => 4
      block = lambda { pr b === Sqrt(a) + Sqrt(b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm256_cvtpd_ps( _mm256_add_pd( _mm256_sqrt_pd( a ), _mm256_cvtps_pd( _mm_sqrt_ps( b ) ) ) );
EOF
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
      push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_raises( IntrinsicsError, &block )
      }
    }
  end

  def test_add_int_simple
    push_env( :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 4
      b = Int  :b, :vector_length => 4, :size => 2
      block = lambda { pr a + b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_ps( a, _mm_cvtepi32_ps( _mm_cvtepi16_epi32( b ) ) );
EOF
      push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
vaddq_f32( a, vcvtq_f32_s32( vmovl_s16( b ) ) );
EOF
      }
    }
  end

  def test_add_int16_double_skylake_avx512
    push_env( :default_real_size => 8, :default_int_size => 2, :lang => C, :model => "skylake-avx512", :architecture => X86 ) {
      a = Real :a, :vector_length => 2
      b = Int  :b, :vector_length => 2
      block = lambda { pr a + b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_pd( a, _mm_cvtepi64_pd( _mm_cvtepi16_epi64( b ) ) );
EOF
      push_env( :architecture => ARM ) {
        assert_raises( IntrinsicsError, &block )
      }
    }
  end

  def test_add_int_double
    push_env( :default_real_size => 8, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 2
      b = Int  :b, :vector_length => 2
      block = lambda { pr a + b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_pd( a, _mm_cvtpi32_pd( b ) );
EOF
      push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_raises( IntrinsicsError, &block )
      }
      push_env( :architecture => ARM, :model => "armv8-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
vaddq_f64( a, vcvt_f64_f32( vcvt_f32_s32( b ) ) );
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
      push_env( :architecture => ARM, :model => "armv7-a" ) {
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
      push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
vmulq_f32( a, b );
EOF
      }
    }
  end

  def test_mul_double
    push_env( :default_real_size => 8, :lang => C, :model => :nehalem, :architecture => X86 ) {
      a = Real :a, :vector_length => 2
      b = Real :b, :vector_length => 2
      block = lambda { pr a * b }
      assert_subprocess_output( <<EOF, "", &block )
_mm_mul_pd( a, b );
EOF
      push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_raises( IntrinsicsError, &block )
      }
      push_env( :architecture => ARM, :model => "armv8-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
vmulq_f64( a, b );
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
      push_env( :architecture => ARM, :model => "armv7-a" ) {
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
      block2 = lambda { pr FMA(a,b,1.5) }
      assert_subprocess_output( <<EOF, "", &block )
_mm_add_ps( c, _mm_mul_ps( a, b ) );
EOF
      assert_subprocess_output( <<EOF, "", &block2 )
_mm_add_ps( _mm_set1_ps( 1.5f ), _mm_mul_ps( a, b ) );
EOF
      push_env( :model => :haswell ) {
        assert_subprocess_output( <<EOF, "", &block )
_mm_fmadd_ps( a, b, c );
EOF
        assert_subprocess_output( <<EOF, "", &block2 )
_mm_fmadd_ps( a, b, _mm_set1_ps( 1.5f ) );
EOF
      }
      push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
vmlaq_f32( c, a, b );
EOF
        assert_subprocess_output( <<EOF, "", &block2 )
vmlaq_f32( vdupq_n_f32( 1.5f ), a, b );
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
      push_env( :architecture => ARM, :model => "armv7-a" ) {
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
        push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
b = vld1q_f32( &a[0] );
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
        push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
b = vld1q_f32( &a[0] );
EOF
        }
      }
    }
  end

  def test_mask_load_old
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :sandybridge, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block = lambda { pr b === MaskLoad(a[0], [1, 0, 1, 0], b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_maskload_ps( (float * ) &a[0], _mm_setr_epi32( -1, 0, -1, 0 ) );
EOF
    }
  end

  def test_zeromask_load_old
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :sandybridge, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block = lambda { pr b === MaskLoad(a[0], [0, 0, 0, 0], b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_setzero_ps( );
EOF
    }
  end

  def test_fullmask_load_old
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
        push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
vst1q_f32( (float * ) &a[0], b );
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
        push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
vst1q_f32( (float * ) &a[0], vaddq_f32( b, b ) );
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
        push_env( :architecture => ARM, :model => "armv7-a" ) {
        assert_subprocess_output( <<EOF, "", &block )
vst1q_f32( (float * ) &a[0], b );
EOF
        }
      }
    }
  end

  def test_mask_store_old
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :sandybridge, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block = lambda { pr MaskStore(a[0], b, [1, 0, 1, 0]) }
      assert_subprocess_output( <<EOF, "", &block )
_mm_maskstore_ps( (float * ) &a[0], _mm_setr_epi32( -1, 0, -1, 0 ), b );
EOF
    }
  end

  def test_fullmask_store_old
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :sandybridge, :architecture => X86 ) {
      a = Real :a, :dim => [Dim()]
      b = Real :b, :vector_length => 4
      block = lambda { pr MaskStore(a[0], b, [1, 1, 1, 1]) }
      assert_subprocess_output( <<EOF, "", &block )
_mm_storeu_ps( (float * ) &a[0], b );
EOF
    }
  end

  def test_zeromask_store_old
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
b = _mm_set1_ps( 1.0f );
EOF
      push_env( :architecture => ARM, :model => "armv7-a" ) {
      assert_subprocess_output( <<EOF, "", &block )
b = vdupq_n_f32( 1.0f );
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
      push_env( :architecture => ARM, :model => "armv7-a" ) {
      assert_subprocess_output( <<EOF, "", &block )
b = vdupq_n_f32( 0.0f );
EOF
      }
    }
  end

  def test_set
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      b = Real :b, :vector_length => 4
      block = lambda { pr b === Set([1.0, 2.0, 3.0, 4.0], b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_setr_ps( 1.0f, 2.0f, 3.0f, 4.0f );
EOF
      push_env( :architecture => ARM, :model => "armv7-a" ) {
      assert_subprocess_output( <<EOF, "", &block )
b = vsetq_lane_f32( 4.0f, vsetq_lane_f32( 3.0f, vsetq_lane_f32( 2.0f, vsetq_lane_f32( 1.0f, vdupq_n_f32( 0 ), 0 ), 1 ), 2 ), 3 );
EOF
      }
    }
  end

  def test_set_identical
    push_env(:array_start => 0, :default_real_size => 4, :lang => C, :model => :nehalem, :architecture => X86 ) {
      b = Real :b, :vector_length => 4
      block = lambda { pr b === Set([1.0, 1.0, 1.0, 1.0], b) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm_set1_ps( 1.0f );
EOF
      push_env( :architecture => ARM, :model => "armv7-a" ) {
      assert_subprocess_output( <<EOF, "", &block )
b = vdupq_n_f32( 1.0f );
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
      push_env( :architecture => ARM, :model => "armv7-a" ) {
      assert_subprocess_output( <<EOF, "", &block )
b = vdupq_n_f32( 0.0f );
EOF
      }
    }
  end

  def test_mask
    push_env(:array_start => 0, :lang => C, :model => :knl, :architecture => X86 ) {
      m = Mask([1,1,0,1])
      block = lambda { pr Expression( Addition, m, m ) }
      assert_subprocess_output( <<EOF, "", &block )
0x1011 + 0x1011;
EOF
      assert_subprocess_output( <<EOF, "" ) { decl m.value }
const uint8_t 0x1011 = 0x1011;
EOF
      m = Mask([1,0,0,1], :length => 4)
      assert_subprocess_output( <<EOF, "", &block )
0x1001 + 0x1001;
EOF
      assert_kind_of(Variable, m.value)
      assert_kind_of(Int, m.value.type)
      assert_equal(1, m.value.type.size)
      assert_raises(OperatorError) {
        m = Mask([1,0,0,1], :length => 6)
      }
      m = Mask( Int(:a, :size => 1), :length => 6)
      assert_subprocess_output( <<EOF, "", &block )
a + a;
EOF
      assert_raises(OperatorError) {
        m = Mask( Int(:a, :size => 1), :length => 9 )
      }
    }
  end

  def test_mask_load
    push_env(:array_start => 0, :lang => C, :model => "skylake-avx512".to_sym, :architecture => X86 ) {
      m = Mask([1,1,0,1])
      a = Real(:a, :dim => Dim())
      b = Real(:b, :vector_length => 8)
      block = lambda { pr b === Load(a[0], b, :mask => m ) }
      assert_raises(OperatorError, &block)
      b = Real(:b, :vector_length => 4)
      assert_subprocess_output( <<EOF, "", &block )
b = _mm256_mask_loadu_pd( b, (uint8_t)0x1011, &a[0] );
EOF
      push_env( :model => :ivybridge ) {
        assert_raises(IntrinsicsError, &block)
      }
    }
  end

  def test_maskz_load
    push_env(:array_start => 0, :lang => C, :model => "skylake-avx512".to_sym, :architecture => X86 ) {
      m = Mask([1,1,0,1])
      a = Real(:a, :dim => Dim())
      b = Real(:b, :vector_length => 4)
      block = lambda { pr b === Load(a[0], b, :mask => m, :zero => true ) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm256_maskz_loadu_pd( (uint8_t)0x1011, &a[0] );
EOF
    }
  end

  def test_full_mask_load
    push_env(:array_start => 0, :lang => C, :model => "skylake-avx512".to_sym, :architecture => X86 ) {
      m = Mask([1,1,1,1])
      a = Real(:a, :dim => Dim())
      b = Real(:b, :vector_length => 8)
      block = lambda { pr b === Load(a[0], b, :mask => m ) }
      assert_raises(OperatorError, &block)
      b = Real(:b, :vector_length => 4)
      assert_subprocess_output( <<EOF, "", &block )
b = _mm256_loadu_pd( &a[0] );
EOF
      push_env( :model => :ivybridge ) {
        assert_subprocess_output( <<EOF, "", &block )
b = _mm256_loadu_pd( &a[0] );
EOF
      }
    }
  end

  def test_full_maskz_load
    push_env(:array_start => 0, :lang => C, :model => "skylake-avx512".to_sym, :architecture => X86 ) {
      m = Mask([1,1,1,1])
      a = Real(:a, :dim => Dim())
      b = Real(:b, :vector_length => 4)
      block = lambda { pr b === Load(a[0], b, :mask => m, :zero => true ) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm256_loadu_pd( &a[0] );
EOF
    }
  end

  def test_empty_mask_load
    push_env(:array_start => 0, :lang => C, :model => "skylake-avx512".to_sym, :architecture => X86 ) {
      m = Mask([0,0,0,0])
      a = Real(:a, :dim => Dim())
      b = Real(:b, :vector_length => 8)
      block = lambda { pr b === Load(a[0], b, :mask => m ) }
      assert_raises(OperatorError, &block)
      b = Real(:b, :vector_length => 4)
      assert_subprocess_output( <<EOF, "", &block )
b = b;
EOF
      push_env( :model => :ivybridge ) {
        assert_subprocess_output( <<EOF, "", &block )
b = b;
EOF
      }
    }
  end

  def test_empty_maskz_load
    push_env(:array_start => 0, :lang => C, :model => "skylake-avx512".to_sym, :architecture => X86 ) {
      m = Mask([0,0,0,0])
      a = Real(:a, :dim => Dim())
      b = Real(:b, :vector_length => 4)
      block = lambda { pr b === Load(a[0], b, :mask => m, :zero => true ) }
      assert_subprocess_output( <<EOF, "", &block )
b = _mm256_setzero_pd( );
EOF
    }
  end

  def test_mask_store
    push_env(:array_start => 0, :lang => C, :model => "skylake-avx512".to_sym, :architecture => X86 ) {
      m = Mask([1,1,0,1])
      a = Real(:a, :dim => Dim())
      b = Real(:b, :vector_length => 4)
      block = lambda { pr Store(a[0], b, :mask => m ) }
      assert_subprocess_output( <<EOF, "", &block )
_mm256_mask_storeu_pd( (double * ) &a[0], (uint8_t)0x1011, b );
EOF
    }
  end

  def test_full_mask_store
    push_env(:array_start => 0, :lang => C, :model => "skylake-avx512".to_sym, :architecture => X86 ) {
      m = Mask([1,1,1,1])
      a = Real(:a, :dim => Dim())
      b = Real(:b, :vector_length => 4)
      block = lambda { pr Store(a[0], b, :mask => m ) }
      assert_subprocess_output( <<EOF, "", &block )
_mm256_storeu_pd( (double * ) &a[0], b );
EOF
    }
  end

  def test_empty_mask_store
    push_env(:array_start => 0, :lang => C, :model => "skylake-avx512".to_sym, :architecture => X86 ) {
      m = Mask([0,0,0,0])
      a = Real(:a, :dim => Dim())
      b = Real(:b, :vector_length => 4)
      block = lambda { pr Store(a[0], b, :mask => m ) }
      assert_subprocess_output( <<EOF, "", &block )
;
EOF
    }
  end

end
