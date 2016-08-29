module BOAST

  # @private
  module TimerProbe
    extend PrivateStateAccessor

    RESULT = BOAST::Int( "_boast_duration", :size => 8 )

    module_function

    def header
      if OS.mac? then
        get_output.print <<EOF
#if __cplusplus
extern "C" {
#endif
#include <mach/mach_time.h>
#if __cplusplus
}
#endif
EOF
      else
        get_output.print "#include <time.h>\n"
      end
    end

    def decl
      if OS.mac? then
        get_output.print "  uint64_t _mac_boast_start, _mac_boast_stop;\n"
        get_output.print "  mach_timebase_info_data_t _mac_boast_timebase_info;\n"
      else
        get_output.print "  struct timespec _boast_start, _boast_stop;\n"
      end
      BOAST::decl RESULT
    end

    def configure
    end

    def start
      if OS.mac? then
        get_output.print "  _mac_boast_start = mach_absolute_time();\n"
      else
        get_output.print "  clock_gettime(CLOCK_REALTIME, &_boast_start);\n"
      end
    end

    def stop
      if OS.mac? then
        get_output.print "  _mac_boast_stop = mach_absolute_time();\n"
      else
        get_output.print "  clock_gettime(CLOCK_REALTIME, &_boast_stop);\n"
      end
    end

    def compute
      if OS.mac? then
        get_output.print "  mach_timebase_info(&_mac_boast_timebase_info);\n"
        get_output.print "  #{RESULT} = (_mac_boast_stop - _mac_boast_start) * _mac_boast_timebase_info.numer / _mac_boast_timebase_info.denom;\n"
      else
        get_output.print "  #{RESULT} = (int64_t)(_boast_stop.tv_sec - _boast_start.tv_sec) * 1000000000ll + _boast_stop.tv_nsec - _boast_start.tv_nsec;\n"
      end
      get_output.print "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"duration\")),rb_float_new((double)#{RESULT}*(double)1e-9));\n"
    end

    def get_options
    end

  end

  # @private
  module PAPIProbe
    extend PrivateStateAccessor

    module_function

    def name
      return "PAPI"
    end

    def header
    end

    def decl
      get_output.print "  VALUE _boast_event_set = Qnil;\n"
      get_output.print "  VALUE _boast_papi_results = Qnil;\n"
    end

    def configure
      get_output.print <<EOF
  if( _boast_rb_opts != Qnil ) {
     VALUE _boast_PAPI_rb_ptr = Qnil;
    _boast_PAPI_rb_ptr = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("PAPI")));
    if( _boast_PAPI_rb_ptr != Qnil ) {
      VALUE _boast_PAPI = Qnil;
      VALUE _boast_EventSet = Qnil;
      rb_eval_string("require 'PAPI'");
      _boast_PAPI =  rb_const_get(rb_cObject, rb_intern("PAPI"));
      _boast_EventSet =  rb_const_get(_boast_PAPI, rb_intern("EventSet"));
      _boast_event_set = rb_funcall(_boast_EventSet, rb_intern("new"), 0);
      rb_funcall(_boast_event_set, rb_intern("add_named"), 1, _boast_PAPI_rb_ptr);
    }
  }
EOF
    end

    def start
      get_output.print <<EOF
  if( _boast_event_set != Qnil) {
    rb_funcall(_boast_event_set, rb_intern("start"), 0);
  }
EOF
  end

    def stop
      get_output.print <<EOF
  if( _boast_event_set != Qnil) {
    _boast_papi_results = rb_funcall(_boast_event_set, rb_intern("stop"), 0);
  }
EOF
    end

    def compute
      get_output.print <<EOF
  if( _boast_papi_results != Qnil) {
    VALUE _boast_papi_stats = Qnil;
    _boast_papi_stats = rb_ary_new3(1,rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("PAPI"))));
    _boast_papi_stats = rb_funcall(_boast_papi_stats, rb_intern("flatten"), 0);
    _boast_papi_stats = rb_funcall(_boast_papi_stats, rb_intern("zip"), 1, _boast_papi_results);
    _boast_papi_results = rb_funcall(rb_const_get(rb_cObject, rb_intern("Hash")), rb_intern("send"), 2, ID2SYM(rb_intern("[]")), _boast_papi_stats );
    rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"PAPI\")),_boast_papi_results);
  }
EOF
    end

    def get_options
    end

  end

end

