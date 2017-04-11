module BOAST

  # @private
  module HwlocProbe
    extend PrivateStateAccessor

    class << self
      attr_accessor :topology
    end
    @topology = nil

    begin
      require 'hwloc'
      @topology = Hwloc::Topology::new
      @topology.load
    rescue
    end

    module_function

    def cflags
    end

    def header
    end

    def preamble
    end


    def decl
    end

    def configure
    end

    def start
    end

    def stop
    end

    def compute
    end

    def store
    end

    def is_available?
      return  !@topology.nil?
    end

  end

  # @private
  module PthreadAffinityProbe
    extend PrivateStateAccessor

    module_function

    def cflags
      return "-D_GNU_SOURCE"
    end

    def header
      get_output.puts "#include <sched.h>"
    end

    def preamble
      get_output.puts <<EOF
static int _boast_affinity_setup( VALUE _boast_rb_opts, cpu_set_t * _boast_affinity_mask_old );
static int _boast_affinity_setup( VALUE _boast_rb_opts, cpu_set_t * _boast_affinity_mask_old ) {
  if( _boast_rb_opts != Qnil ) {
    VALUE _boast_affinity_rb_ptr = Qnil;

    _boast_affinity_rb_ptr = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("cpu_affinity")));

    if( _boast_affinity_rb_ptr != Qnil ) {
      cpu_set_t _boast_affinity_mask;
      int _boast_affinity_counter;
      int _boast_affinity_cpu_number;

      if( TYPE(_boast_affinity_rb_ptr) != T_ARRAY ) {
        rb_raise(rb_eArgError, "Option :cpu_affinity should be an array!");
      }
      CPU_ZERO(&_boast_affinity_mask);
      _boast_affinity_cpu_number = RARRAY_LEN(_boast_affinity_rb_ptr);
      for( _boast_affinity_counter = 0; _boast_affinity_counter < _boast_affinity_cpu_number; _boast_affinity_counter++ ) {
        CPU_SET(FIX2INT(rb_ary_entry(_boast_affinity_rb_ptr,_boast_affinity_counter)), &_boast_affinity_mask);
      }
      pthread_getaffinity_np(pthread_self(), sizeof(*_boast_affinity_mask_old), _boast_affinity_mask_old);
      if( pthread_setaffinity_np(pthread_self(), sizeof(_boast_affinity_mask), &_boast_affinity_mask) != 0) {
        rb_raise(rb_eArgError, "Invalid affinity list provided!");
      }
      return 1;
    }
  }
  return 0;
}

static int _boast_restore_affinity( int _boast_affinity_set, cpu_set_t * _boast_affinity_mask_old );
static int _boast_restore_affinity( int _boast_affinity_set, cpu_set_t * _boast_affinity_mask_old ){
  if ( _boast_affinity_set == 1 ) {
    pthread_setaffinity_np(pthread_self(), sizeof(*_boast_affinity_mask_old), _boast_affinity_mask_old);
  }
  return 0;
}

EOF
    end

    def decl
      get_output.puts "  cpu_set_t _boast_affinity_mask_old;"
      get_output.puts "  int _boast_affinity_set;"
    end

    def configure
      get_output.print <<EOF
  _boast_affinity_set = _boast_affinity_setup( _boast_rb_opts, &_boast_affinity_mask_old);
EOF
    end

    def start
    end

    def stop
    end

    def compute
      get_output.print <<EOF
  _boast_affinity_set = _boast_restore_affinity( _boast_affinity_set, &_boast_affinity_mask_old);
EOF
    end

    def store
    end

    def is_available?
      return false if OS.mac?
      return true
    end

  end

  if HwlocProbe.is_available?
    AffinityProbe = HwlocProbe
  elsif PthreadAffinityProbe.is_available?
    AffinityProbe = PthreadAffinityProbe
  else
    AffinityProbe = nil
  end

end
