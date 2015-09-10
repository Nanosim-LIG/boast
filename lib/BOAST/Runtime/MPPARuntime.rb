module BOAST

  module MPPAProbe
    extend PrivateStateAccessor

    module_function

    # Monitoring has to be started before transfer begin and after transfers end as we don't have sync points in between.
    def header
      get_output.puts "#include <mppa_mon.h>"
    end

    def decl
      get_output.print <<EOF
  float _mppa_avg_pwr;
  float _mppa_energy;
  float _mppa_duration;
  mppa_mon_ctx_t * _mppa_ctx;
  mppa_mon_sensor_t _mppa_pwr_sensor[] = {MPPA_MON_PWR_MPPA0};
  mppa_mon_measure_report_t * _mppa_report;
  mppa_mon_open(0, &_mppa_ctx);
  mppa_mon_measure_set_sensors(_mppa_ctx, _mppa_pwr_sensor, 1);
  mppa_mon_measure_start(_mppa_ctx);
EOF
    end

    def configure
    end

    def start
    end

    def stop
    end

    def compute
      get_output.print <<EOF
  mppa_mon_measure_stop(_mppa_ctx, &_mppa_report);
  _mppa_avg_pwr = 0;
  _mppa_energy = 0;
  for(_mppa_i=0; _mppa_i < _mppa_report->count; _mppa_i++){
    _mppa_avg_pwr += _mppa_report->measures[_mppa_i].avg_power;
    _mppa_energy += _mppa_report->measures[_mppa_i].total_energy;
  }
  _mppa_avg_pwr = _mppa_avg_pwr/(float) _mppa_report->count;
  _mppa_duration = _mppa_report->total_time;
  mppa_mon_measure_free_report(_mppa_report);
  mppa_mon_close(_mppa_ctx);
  rb_hash_aset(_boast_stats,ID2SYM(rb_intern("mppa_avg_pwr")),rb_float_new(_mppa_avg_pwr));
  rb_hash_aset(_boast_stats,ID2SYM(rb_intern("mppa_energy")),rb_float_new(_mppa_energy));
  rb_hash_aset(_boast_stats,ID2SYM(rb_intern("mppa_duration")), rb_float_new(_mppa_duration));
EOF
    end

  end

  module MPPARuntime
    include CRuntime

    alias create_targets_old create_targets
    alias cleanup_old cleanup
    alias fill_module_header_old fill_module_header
    alias get_params_value_old get_params_value
    alias fill_decl_module_params_old fill_decl_module_params
    alias get_results_old get_results

    def cleanup(kernel_files)
      cleanup_old(kernel_files)
      ([io_bin, comp_bin, io_object, comp_object]).each { |fn|
        File::unlink(fn)
      }
    end

    def target_depends
      return [ module_file_object ]
    end

    def target_sources
      return [ module_file_source, io_source, comp_source ]
    end

    def multibinary_path
      return "#{base_path}.mpk"
    end

    def io_bin
      return "#{base_path}.binio"
    end

    def comp_bin
      return "#{base_path}.bincomp"
    end

    def io_object
      return "#{base_path}.#{RbConfig::CONFIG["OBJEXT"]}io"
    end

    def comp_object
      return "#{base_path}.#{RbConfig::CONFIG["OBJEXT"]}comp"
    end

    def io_source
      return "#{base_path}.cio"
    end

    def comp_source
      return "#{base_path}.ccomp"
    end

    def set_io
      set_output(@code_io)
    end

    def set_comp
      @code_comp = StringIO::new unless @code_comp
      set_output(@code_comp)
    end

    attr_accessor :code_comp
    attr_accessor :procedure_comp
    attr_accessor :binary_comp
    attr_accessor :multibinary

    def save_binary
      f = File::open(io_object,"rb")
      @binary = StringIO::new
      @binary.write( f.read )
      f.close
      f = File::open(comp_object,"rb")
      @binary_comp = StringIO::new
      @binary_comp.write( f.read )
      f.close
      f = File::open(multibinary_path,"rb")
      @multibinary = StringIO::new
      @multibinary.write( f.read )
      f.close
    end

    def create_targets( linker, ldshared, ldflags, kernel_files )
      create_targets_old( linker, ldshared, ldflags, kernel_files )
      file multibinary_path => [io_bin, comp_bin] do
        sh "k1-create-multibinary --clusters #{comp_bin} --clusters-names \"comp-part\" --boot #{io_bin} --bootname \"io-part\" -T #{multibinary_path}"
      end
      Rake::Task[multibinary_path].invoke
    end

    def fill_multibinary_header
      fill_library_header
      get_output.puts "#include <mppaipc.h>"
      get_output.puts "#include <mppa/osconfig.h>"
      get_output.puts "#include <time.h>"
    end

    def copy_array_param_from_host( param )
      get_output.print <<EOF
  mppa_read(_mppa_from_host_size, &_mppa_#{param}_size, sizeof(_mppa_#{param}_size));
  #{param} = malloc(_mppa_#{param}_size);
  mppa_read(_mppa_from_host_var, #{param}, _mppa_#{param}_size);
EOF
    end

    def copy_scalar_param_from_host( param )
      get_output.print  <<EOF
  mppa_read(_mppa_from_host_var, &#{param}, sizeof(#{param}));
EOF
    end

    def get_cluster_list_from_host
      get_output.print <<EOF
  mppa_read(_mppa_from_host_size, &_mppa_clust_list_size, sizeof(_mppa_clust_list_size));
  _clust_list = malloc(_mppa_clust_list_size);
  _nb_clust = _mppa_clust_list_size / sizeof(*_clust_list);
  mppa_read(_mppa_from_host_var, _clust_list, _mppa_clust_list_size);
EOF
    end

    def copy_array_param_to_host(param)
      get_output.print <<EOF
  mppa_write(_mppa_to_host_var, #{param}, _mppa_#{param}_size);
EOF
    end

    def copy_scalar_param_to_host(param)
      get_output.print <<EOF
  mppa_write(_mppa_to_host_var, &#{param}, sizeof(#{param}));
EOF
    end

    def multibinary_main_io_source_decl
      #Parameters declaration
      @procedure.parameters.each { |param|
        get_output.print "  #{param.type.decl} "
        get_output.print "*" if param.dimension or param.scalar_output?
        get_output.puts "#{param.name};"
        if param.dimension then
          get_output.puts "  size_t _mppa_#{param}_size;"
        end
      }

      #Return value declaration
      get_output.puts "  #{@procedure.properties[:return].type.decl} _mppa_ret;" if @procedure.properties[:return]

      #Cluster list declaration
      get_output.print <<EOF
  uint32_t *_clust_list;
  int _nb_clust;
  int _mppa_clust_list_size;
EOF

      #Timer
      get_output.print <<EOF
  struct timespec _mppa_start, _mppa_stop;
  int64_t _mppa_duration;
EOF

      #Communication variables
      get_output.print <<EOF
  int _mppa_from_host_size, _mppa_from_host_var;
  int _mppa_to_host_size,   _mppa_to_host_var;
  int _mppa_pid[16], _mppa_i;
EOF
    end

    def multibinary_main_io_source_get_params
      #Receiving parameters from Host
      get_output.print <<EOF
  _mppa_from_host_size = mppa_open("/mppa/buffer/board0#mppa0#pcie0#2/host#2", O_RDONLY);
  _mppa_from_host_var = mppa_open("/mppa/buffer/board0#mppa0#pcie0#3/host#3", O_RDONLY);
EOF
      @procedure.parameters.each { |param|
        if param.dimension then
          copy_array_param_from_host(param)
        else
          copy_scalar_param_from_host(param)
        end
      }

      #Receiving cluster list
      get_cluster_list_from_host

      get_output.print <<EOF
  mppa_close(_mppa_from_host_size);
  mppa_close(_mppa_from_host_var);
EOF
    end

    def multibinary_main_io_source_send_results
      #Sending results to Host
      get_output.print <<EOF
  _mppa_to_host_var = mppa_open("/mppa/buffer/host#4/board0#mppa0#pcie0#4", O_WRONLY);
EOF
      @procedure.parameters.each { |param| 
        if param.direction == :out or param.direction == :inout then
          if param.dimension then
            copy_array_param_to_host(param)
          else
            copy_scalar_param_to_host(param)
          end
        end
      }
      copy_scalar_param_to_host("_mppa_ret") if @procedure.properties[:return]
      copy_scalar_param_to_host("_mppa_duration")
      get_output.print <<EOF
  mppa_close(_mppa_to_host_var);
EOF
    end

    def fill_multibinary_main_io_source
      multibinary_main_io_source_decl

      multibinary_main_io_source_get_params

      #Spawning cluster
      get_output.print <<EOF

  clock_gettime(CLOCK_REALTIME, &_mppa_start);
  for(_mppa_i=0; _mppa_i<_nb_clust; _mppa_i++){
    _mppa_pid[_mppa_i] = mppa_spawn(_clust_list[_mppa_i], NULL, "comp-part", NULL, NULL);
  }
EOF
      #Calling IO procedure
      get_output.print "  _mppa_ret =" if @procedure.properties[:return]
      get_output.print "  #{@procedure.name}("
      get_output.print @procedure.parameters.map(&:name).join(", ")
      get_output.puts ");"

      #Waiting for clusters
      get_output.print <<EOF
  for(_mppa_i=0; _mppa_i<_nb_clust; _mppa_i++){
    mppa_waitpid(_mppa_pid[_mppa_i], NULL, 0);
  }
  clock_gettime(CLOCK_REALTIME, &_mppa_stop);
  _mppa_duration = (_mppa_stop.tv_sec - _mppa_start.tv_sec) * (unsigned long long int)1000000000 + _mppa_stop.tv_nsec - _mppa_start.tv_nsec;
EOF

      multibinary_main_io_source_send_results
    end

    def fill_multibinary_main_comp_source
      if @procedure_comp then
        get_output.puts "    #{@procedure_comp.name}();"
      end
    end

    def fill_multibinary_source(mode)
      fill_multibinary_header
      code = nil
      if mode == :io then
        code = @code
      else
        code = @code_comp
      end
      if code then
        code.rewind
        get_output.print code.read
      end
      get_output.puts "int main(int argc, const char* argv[]) {"
      if mode == :io then
        fill_multibinary_main_io_source
      else
        fill_multibinary_main_comp_source
      end
      get_output.print <<EOF
  mppa_exit(0);
  return 0;
}
EOF
    end

    def create_multibinary_source(mode)
      f = File::open(self.send("#{mode}_source"),"w+")
      previous_lang = get_lang
      previous_output = get_output
      set_output(f)
      set_lang(@lang)

      fill_multibinary_source(mode)

      if debug_source? then
        f.rewind
        puts f.read
      end
      set_output(previous_output)
      set_lang(previous_lang)
      f.close
    end

    def create_multibinary_sources
      create_multibinary_source(:io)
      create_multibinary_source(:comp)
    end

    def create_sources
      create_multibinary_sources
      create_module_file_source
    end

    def fill_module_header
      fill_module_header_old
      get_output.puts "#include <mppaipc.h>"
    end

    def fill_decl_module_params
      fill_decl_module_params_old
      get_output.print <<EOF
  int _mppa_i;
  int _mppa_load_id;
  int _mppa_pid;
  int _mppa_fd_size;
  int _mppa_fd_var;
  int _mppa_clust_list_size;
  int _mppa_clust_nb;
  uint32_t * _mppa_clust_list;
  _mppa_load_id = mppa_load(0, 0, 0, \"#{multibinary_path}\");
  _mppa_pid = mppa_spawn(_mppa_load_id, NULL, \"io-part\", NULL, NULL);
EOF
    end

    def copy_array_param_from_ruby( param, ruby_param )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      (rb_ptr === ruby_param).pr
      get_output.print <<EOF
  if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    size_t _boast_array_size;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    _boast_array_size = _boast_n_ary->total * na_sizeof[_boast_n_ary->type];
    mppa_write(_mppa_fd_size, &_boast_array_size, sizeof(_boast_array_size));
    #{param} = (void *) _boast_n_ary->ptr;
    mppa_write(_mppa_fd_var, #{param}, _boast_array_size);
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for %s, expecting array!", "#{param}");
  }
EOF
    end

    def copy_scalar_param_from_ruby( param, ruby_param )
      case param.type
      when Int
        (param === FuncCall::new("NUM2INT", ruby_param)).pr if param.type.size == 4
        (param === FuncCall::new("NUM2LONG", ruby_param)).pr if param.type.size == 8
      when Real
        (param === FuncCall::new("NUM2DBL", ruby_param)).pr
      end
      get_output.puts "  mppa_write(_mppa_fd_var, &#{param}, sizeof(#{param}));"
    end

    def get_params_value
      get_output.print <<EOF
  _mppa_fd_size = mppa_open(\"/mppa/buffer/board0#mppa0#pcie0#2/host#2\", O_WRONLY);
  _mppa_fd_var = mppa_open(\"/mppa/buffer/board0#mppa0#pcie0#3/host#3\", O_WRONLY);
EOF
      get_params_value_old
      get_output.print <<EOF
  if(_boast_rb_opts != Qnil) {
    _boast_rb_ptr = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("clusters")));
    if (_boast_rb_ptr != Qnil ) {
      int _boast_i;
      _mppa_clust_nb = RARRAY_LEN(_boast_rb_ptr);
      _mppa_clust_list = malloc(sizeof(uint32_t)*_mppa_clust_nb);
      for(_boast_i=0; _boast_i < _mppa_clust_nb; _boast_i++){
        _mppa_clust_list[_boast_i] = NUM2INT(rb_ary_entry(_boast_rb_ptr, _boast_i));
      }
    } else {
      _mppa_clust_list = malloc(sizeof(uint32_t));
      _mppa_clust_list[0] = 0;
      _mppa_clust_nb = 1;
    }
  } else {
    _mppa_clust_list = malloc(sizeof(uint32_t));
    _mppa_clust_list[0] = 0;
    _mppa_clust_nb = 1;
  }
  
  _mppa_clust_list_size = sizeof(uint32_t)*_mppa_clust_nb;
  mppa_write(_mppa_fd_size, &_mppa_clust_list_size, sizeof(_mppa_clust_list_size));
  mppa_write(_mppa_fd_var, _mppa_clust_list, _mppa_clust_list_size);
  free(_mppa_clust_list);
  mppa_close(_mppa_fd_var);
  mppa_close(_mppa_fd_size);
EOF
    end

    def create_procedure_call
    end

    def copy_array_param_to_ruby(param, ruby_param)
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      (rb_ptr === ruby_param).pr
      get_output.print <<EOF
  if ( IsNArray(_boast_rb_ptr) ) {
EOF
      if param.direction == :out or param.direction == :inout then
        get_output.print <<EOF
    struct NARRAY *_boast_n_ary;
    size_t _boast_array_size;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    _boast_array_size = _boast_n_ary->total * na_sizeof[_boast_n_ary->type];
    mppa_read(_mppa_fd_var, #{param}, _boast_array_size);
EOF
      end
        get_output.print <<EOF
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for %s, expecting array!", "#{param}");
  }
EOF
    end

    def copy_scalar_param_to_ruby(param, ruby_param)
      if param.scalar_output? then
        get_output.print <<EOF
  mppa_read(_mppa_fd_var, &#{param}, sizeof(#{param}));
EOF
        case param.type
        when Int
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_int_new((long long)#{param}));" if param.type.signed?
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_int_new((unsigned long long)#{param}));" if not param.type.signed?
        when Real
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_float_new((double)#{param}));"
        end
      end
    end

    def get_results
      get_output.print <<EOF
  _mppa_fd_var = mppa_open(\"/mppa/buffer/host#4/board0#mppa0#pcie0#4\", O_RDONLY);
EOF
      get_results_old
      get_output.puts "  mppa_read(_mppa_fd_var, &_boast_ret, sizeof(_boast_ret));" if @procedure.properties[:return]
      get_output.puts "  mppa_read(_mppa_fd_var, &_boast_duration, sizeof(_boast_duration));"
      get_output.print <<EOF
  mppa_close(_mppa_fd_var);
  mppa_waitpid(_mppa_pid, NULL, 0);
  mppa_unload(_mppa_load_id);
EOF
    end

  end

end
