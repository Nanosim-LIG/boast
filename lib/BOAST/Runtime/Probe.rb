module BOAST

  # @private
  module TimerProbe
    extend PrivateStateAccessor

    RESULT = BOAST::Int( "_boast_duration", :size => 8 )

    module_function

    def ldflags
      if OS.mac? then
        ""
      else
        "-lrt"
      end
    end

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

    def preamble
      get_output.print <<EOF
struct _boast_timer_struct {
EOF
      if OS.mac? then
        get_output.print <<EOF
  uint64_t start, stop;
  mach_timebase_info_data_t timebase_info;
EOF
      else
        get_output.print <<EOF
  struct timespec start, stop;
EOF
      end
      push_env(:indent_level => 2) {
        BOAST::decl RESULT
      }
      get_output.print <<EOF
};

static inline void _boast_timer_start(struct _boast_timer_struct * _boast_timer) {
EOF
      if OS.mac? then
        get_output.print "  _boast_timer->start = mach_absolute_time();\n"
      else
        get_output.print "  clock_gettime(CLOCK_REALTIME, &_boast_timer->start);\n"
      end
      get_output.print <<EOF
}

static inline void _boast_timer_stop(struct _boast_timer_struct * _boast_timer) {
EOF
      if OS.mac? then
        get_output.print "  _boast_timer->stop = mach_absolute_time();\n"
      else
        get_output.print "  clock_gettime(CLOCK_REALTIME, &_boast_timer->stop);\n"
      end
      get_output.print <<EOF
}

static inline void _boast_timer_compute(struct _boast_timer_struct * _boast_timer) {
EOF
      if OS.mac? then
        get_output.print "  mach_timebase_info(&_boast_timer->timebase_info);\n"
        get_output.print "  _boast_timer->#{RESULT} = (_boast_timer->stop - _boast_timer->start) * _boast_timer->timebase_info.numer / _boast_timer->timebase_info.denom;\n"
      else
        get_output.print "  _boast_timer->#{RESULT} = (int64_t)(_boast_timer->stop.tv_sec - _boast_timer->start.tv_sec) * 1000000000ll + _boast_timer->stop.tv_nsec - _boast_timer->start.tv_nsec;\n"
      end
      get_output.print <<EOF
}

#ifdef RUBY
static inline void _boast_timer_store(struct _boast_timer_struct * _boast_timer, VALUE _boast_stats) {
EOF
      get_output.print "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"duration\")),rb_float_new((double)_boast_timer->#{RESULT}*(double)1e-9));\n"
      if OS.mac? then
        get_output.print "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"start\")),rb_int_new((int64_t)(_boast_timer->start * _boast_timer->timebase_info.numer / _boast_timer->timebase_info.denom)*1000000000ll));\n"
        get_output.print "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"end\")),rb_int_new((int64_t)(_boast_timer->stop * _boast_timer->timebase_info.numer / _boast_timer->timebase_info.denom)*1000000000ll));\n"
      else
        get_output.print "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"start\")),rb_int_new((int64_t)_boast_timer->start.tv_sec * 1000000000ll+_boast_timer->start.tv_nsec));\n"
        get_output.print "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"end\")),rb_int_new((int64_t)_boast_timer->stop.tv_sec * 1000000000ll+_boast_timer->stop.tv_nsec));\n"
      end
      get_output.print <<EOF
}
#endif

EOF
    end

    def decl
      get_output.print "  struct _boast_timer_struct _boast_timer;\n"
    end

    def configure
    end

    def start
      get_output.puts "  _boast_timer_start(&_boast_params._boast_timer);"
    end

    def stop
      get_output.puts "  _boast_timer_stop(&_boast_params._boast_timer);"
    end

    def compute
      get_output.puts "  _boast_timer_compute(&_boast_params._boast_timer);"
    end

    def store
      get_output.puts "  _boast_timer_store(&_boast_params._boast_timer, _boast_stats);"
    end

    def to_yaml
      get_output.print <<EOF
  printf(":duration: %lf\\n", (double)_boast_params._boast_timer.#{RESULT}*(double)1e-9);
EOF
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

    def preamble
      get_output.print <<EOF
struct _boast_papi_struct {
  VALUE event_set;
  VALUE results;
};

static void _boast_get_papi_envent_set( VALUE _boast_rb_opts, struct _boast_papi_struct *_boast_papi );
static void _boast_get_papi_envent_set( VALUE _boast_rb_opts, struct _boast_papi_struct *_boast_papi ) {
  VALUE _boast_event_set = Qnil;
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
  _boast_papi->event_set = _boast_event_set;
}

static void _boast_store_papi_results( struct _boast_papi_struct *_boast_papi, VALUE _boast_rb_opts, VALUE _boast_stats );
static void _boast_store_papi_results( struct _boast_papi_struct *_boast_papi, VALUE _boast_rb_opts, VALUE _boast_stats ) {
  if( _boast_papi->results != Qnil) {
    VALUE _boast_papi_stats = Qnil;
    _boast_papi_stats = rb_ary_new3(1,rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("PAPI"))));
    _boast_papi_stats = rb_funcall(_boast_papi_stats, rb_intern("flatten"), 0);
    _boast_papi_stats = rb_funcall(_boast_papi_stats, rb_intern("zip"), 1, _boast_papi->results);
    _boast_papi->results = rb_funcall(rb_const_get(rb_cObject, rb_intern("Hash")), rb_intern("send"), 2, ID2SYM(rb_intern("[]")), _boast_papi_stats );
    rb_hash_aset(_boast_stats, ID2SYM(rb_intern(\"PAPI\")), _boast_papi->results);
  }
}

EOF
    end

    def decl
      get_output.print <<EOF
  struct _boast_papi_struct _boast_papi = { Qnil, Qnil };
EOF
    end

    def configure
      get_output.print <<EOF
  _boast_get_papi_envent_set( _boast_rb_opts, &_boast_papi );
EOF
    end

    def start
      get_output.print <<EOF
  if( _boast_papi.event_set != Qnil) {
    rb_funcall(_boast_papi.event_set, rb_intern("start"), 0);
  }
EOF
    end

    def stop
      get_output.print <<EOF
  if( _boast_papi.event_set != Qnil) {
    _boast_papi.results = rb_funcall(_boast_papi.event_set, rb_intern("stop"), 0);
  }
EOF
    end

    def compute
    end

    def store
      get_output.print <<EOF
  _boast_store_papi_results( &_boast_papi, _boast_rb_opts, _boast_stats );
EOF
    end

  end

end

