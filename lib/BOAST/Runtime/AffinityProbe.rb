module BOAST

  # @private
  module AffinityProbe
    extend PrivateStateAccessor

    module_function

    def cflags
      return "-D_GNU_SOURCE"
    end

    def header
      get_output.puts "#include <sched.h>"
    end

    def decl
      get_output.puts "  cpu_set_t _boast_affinity_mask_old;"
      get_output.puts "  int _boast_affinity_set = 0;"
    end

    def configure
      get_output.print <<EOF
  if( _boast_rb_opts != Qnil ) {
    VALUE _boast_affinity_rb_ptr = Qnil;

    _boast_affinity_rb_ptr = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("cpu_affinity")));

    if( _boast_affinity_rb_ptr != Qnil ) {
      cpu_set_t _boast_affinity_mask;
      int _boast_affinity_counter;
      int _boast_affinity_cpu_number;

      if( TYPE(_boast_affinity_rb_ptr) != T_ARRAY )
        rb_raise(rb_eArgError, "Option :cpu_affinity should be an array!");
      CPU_ZERO(&_boast_affinity_mask);
      _boast_affinity_cpu_number = RARRAY_LEN(_boast_affinity_rb_ptr);
      for( _boast_affinity_counter = 0; _boast_affinity_counter < _boast_affinity_cpu_number; _boast_affinity_counter++ )
        CPU_SET(FIX2INT(rb_ary_entry(_boast_affinity_rb_ptr,_boast_affinity_counter)), &_boast_affinity_mask);
      sched_getaffinity(getpid(), sizeof(_boast_affinity_mask_old), &_boast_affinity_mask_old);
      if( sched_setaffinity(getpid(), sizeof(_boast_affinity_mask), &_boast_affinity_mask) != 0)
        rb_raise(rb_eArgError, "Invalid affinity list provided!");
      _boast_affinity_set = 1;
    }
  }
EOF
    end

    def start
    end

    def stop
    end

    def compute
      get_output.print <<EOF
  if ( _boast_affinity_set == 1 ) {
    sched_setaffinity(getpid(), sizeof(_boast_affinity_mask_old), &_boast_affinity_mask_old);
    _boast_affinity_set = 0;
  }
EOF
    end

  end

end
