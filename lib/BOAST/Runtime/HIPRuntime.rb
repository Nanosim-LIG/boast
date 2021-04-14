module BOAST

  # @private
  module HIPRuntime
    include CRuntime

    private

    alias fill_library_source_old fill_library_source
    alias fill_library_header_old fill_library_header
    alias fill_module_header_old fill_module_header
    alias get_params_value_old get_params_value
    alias fill_decl_module_params_old fill_decl_module_params
    alias create_procedure_call_parameters_old create_procedure_call_parameters
    alias store_results_old store_results
    alias create_wrapper_old create_wrapper

    def fill_module_header
      fill_module_header_old
      get_output.puts "#include \"hip/hip_runtime.h\" "
      @globals.each { |g|
        get_output.write <<EOF
hipError_t  _boast_send_to_symbol_#{g}(void *p, size_t s);
EOF
      }
    end

    def fill_library_header
      fill_library_header_old
      get_output.puts "#include \"hip/hip_runtime.h\" "
    end

    def fill_library_source
      fill_library_source_old
      get_output.write <<EOF
extern "C" {
  #{@procedure.send(:boast_header_s,HIP)}{
    int _boast_i;
    dim3 dimBlock(_boast_block_size[0], _boast_block_size[1], _boast_block_size[2]);
    dim3 dimGrid(_boast_block_number[0], _boast_block_number[1], _boast_block_number[2]);
    hipEvent_t __start, __stop;
    float __time;
    hipEventCreate(&__start);
    hipEventCreate(&__stop);
    hipEventRecord(__start, 0);
    for( _boast_i = 0; _boast_i < _boast_repeat; _boast_i ++) {
      hipLaunchKernelGGL(#{@procedure.name},dimGrid,dimBlock,0,0,#{@procedure.parameters.join(", ")});
    }
    hipEventRecord(__stop, 0);
    hipEventSynchronize(__stop);
    hipEventElapsedTime(&__time, __start, __stop);
    hipEventDestroy(__start);
    hipEventDestroy(__stop);
    return (unsigned long long int)((double)__time*(double)1e6);
  }
}
EOF
      @globals.each { |g|
        get_output.write <<EOF
extern "C" {
   hipError_t _boast_send_to_symbol_#{g}(void *p, size_t s) {
     return hipMemcpyToSymbol(HIP_SYMBOL(#{g}, p, s);
   }
}
EOF
      }
    end

    def copy_array_param_from_ruby(par, param, ruby_param)
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      (rb_ptr === ruby_param).pr
      get_output.print <<EOF
  if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    size_t _boast_array_size;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    _boast_array_size = _boast_n_ary->total * na_sizeof[_boast_n_ary->type];
    hipError_t err = hipMalloc( (void **) &#{par}, _boast_array_size);
    if (err != hipSuccess)
      rb_raise(rb_eRuntimeError, "Could not allocate hip memory for: %s (%s)!", "#{param}", hipGetErrorName(err));
    err = hipMemcpy(#{par}, (void *) _boast_n_ary->ptr, _boast_array_size, hipMemcpyHostToDevice);
    if (err != hipSuccess)
      rb_raise(rb_eRuntimeError, "Could not copy memory to device for: %s (%s)!", "#{param}", hipGetErrorName(err));
  } else if (TYPE(_boast_rb_ptr) == T_STRING) {
    size_t _boast_array_size = RSTRING_LEN(_boast_rb_ptr);
    hipError_t err = hipMalloc( (void **) &#{par}, _boast_array_size);
    if (err != hipSuccess)
      rb_raise(rb_eRuntimeError, "Could not allocate hip memory for: %s (%s)!", "#{param}", hipGetErrorName(err));
    err = hipMemcpy(#{par}, (void *)RSTRING_PTR(_boast_rb_ptr), _boast_array_size, hipMemcpyHostToDevice);
    if (err != hipSuccess)
      rb_raise(rb_eRuntimeError, "Could not copy memory to device for: %s (%s)!", "#{param}", hipGetErrorName(err));
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for %s, expecting NArray or String!", "#{param}");
  }
EOF
    end

    def define_globals
    end

    def copy_array_global_from_ruby(par, param, ruby_param)
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      (rb_ptr === ruby_param).pr
      get_output.print <<EOF
  if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    size_t _boast_array_size;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    _boast_array_size = _boast_n_ary->total * na_sizeof[_boast_n_ary->type];
    hipError_t err;
    err = _boast_send_to_symbol_#{param}((void *) _boast_n_ary->ptr, _boast_array_size);
    if (err != hipSuccess)
      rb_raise(rb_eRuntimeError, "Could not copy memory to device for: %s (%s)!", "#{param}", hipGetErrorName(err));
  } else if (TYPE(_boast_rb_ptr) == T_STRING) {
    size_t _boast_array_size = RSTRING_LEN(_boast_rb_ptr);
    hipError_t err;
    err = _boast_send_to_symbol_#{param}((void *)RSTRING_PTR(_boast_rb_ptr), _boast_array_size);
    if (err != hipSuccess)
      rb_raise(rb_eRuntimeError, "Could not copy memory to device for: %s (%s)!", "#{param}", hipGetErrorName(err));
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for %s, expecting NArray or String!", "#{param}");
  }
EOF
    end

    def copy_scalar_global_from_ruby(str_par, param, ruby_param )
      str_par_tmp = str_par.copy("_boast_tmp_#{str_par}")
      BOAST.decl str_par_tmp
      case param.type
      when Int
        if param.type.size == 4
          (str_par_tmp === FuncCall::new("NUM2INT", ruby_param)).pr
        elsif param.type.size == 8
          (str_par_tmp === FuncCall::new("NUM2LONG", ruby_param)).pr
        end
      when Real
        (str_par_tmp === FuncCall::new("NUM2DBL", ruby_param)).pr
      when Sizet
        (str_par_tmp === Ternary::new(FuncCall::new("sizeof", "size_t") == 4, FuncCall::new("NUM2INT", ruby_param), FuncCall::new("NUM2LONG", ruby_param))).pr
      else
        raise "Unsupported type as kernel argument:#{param.type}!"
      end
      get_output.print <<EOF
  {
    hipError_t err;
    err = _boast_send_to_symbol_#{param}((void *)&#{str_par_tmp}, sizeof(#{str_par_tmp}));
    if (err != hipSuccess)
      rb_raise(rb_eRuntimeError, "Could not copy memory to device for: %s (%s)!", "#{param}", hipGetErrorName(err));
  }
EOF
    end

    def fill_decl_module_params
      fill_decl_module_params_old
      get_output.print <<EOF
  size_t _boast_block_size[3] = {1,1,1};
  size_t _boast_block_number[3] = {1,1,1};
  int64_t _boast_duration;
EOF
    end

    def get_params_value
      get_params_value_old
      get_output.print <<EOF
  if( _boast_rb_opts != Qnil ) {
    VALUE _boast_rb_array_data = Qnil;
    int _boast_i;
    _boast_rb_ptr = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("block_size")));
    if( _boast_rb_ptr != Qnil ) {
      if (TYPE(_boast_rb_ptr) != T_ARRAY)
        rb_raise(rb_eArgError, "Cuda option block_size should be an array");
      for(_boast_i=0; _boast_i<3; _boast_i++) {
        _boast_rb_array_data = rb_ary_entry(_boast_rb_ptr, _boast_i);
        if( _boast_rb_array_data != Qnil )
          _boast_block_size[_boast_i] = (size_t) NUM2LONG( _boast_rb_array_data );
      }
    } else {
      _boast_rb_ptr = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("local_work_size")));
      if( _boast_rb_ptr != Qnil ) {
        if (TYPE(_boast_rb_ptr) != T_ARRAY)
          rb_raise(rb_eArgError, "Cuda option local_work_size should be an array");
        for(_boast_i=0; _boast_i<3; _boast_i++) {
          _boast_rb_array_data = rb_ary_entry(_boast_rb_ptr, _boast_i);
          if( _boast_rb_array_data != Qnil )
            _boast_block_size[_boast_i] = (size_t) NUM2LONG( _boast_rb_array_data );
        }
      }
    }
    _boast_rb_ptr = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("block_number")));
    if( _boast_rb_ptr != Qnil ) {
      if (TYPE(_boast_rb_ptr) != T_ARRAY)
        rb_raise(rb_eArgError, "Cuda option block_number should be an array");
      for(_boast_i=0; _boast_i<3; _boast_i++) {
        _boast_rb_array_data = rb_ary_entry(_boast_rb_ptr, _boast_i);
        if( _boast_rb_array_data != Qnil )
          _boast_block_number[_boast_i] = (size_t) NUM2LONG( _boast_rb_array_data );
      }
    } else {
      _boast_rb_ptr = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("global_work_size")));
      if( _boast_rb_ptr != Qnil ) {
        if (TYPE(_boast_rb_ptr) != T_ARRAY)
          rb_raise(rb_eArgError, "Cuda option global_work_size should be an array");
        for(_boast_i=0; _boast_i<3; _boast_i++) {
          _boast_rb_array_data = rb_ary_entry(_boast_rb_ptr, _boast_i);
          if( _boast_rb_array_data != Qnil )
            _boast_block_number[_boast_i] = (size_t) NUM2LONG( _boast_rb_array_data ) / _boast_block_size[_boast_i];
        }
      }
    }
  }
EOF
    end

    def create_wrapper
    end

    def link_depend
      File::join(File::dirname(library_object), "link-#{File::basename(library_object)}")
    end

    def cleanup(kernel_files)
      super
      File::unlink(link_depend)
    end

    def create_procedure_call_parameters
      return create_procedure_call_parameters_old + ["_boast_block_number", "_boast_block_size", "_boast_params._boast_repeat"]
    end

    def create_procedure_call
      get_output.print "  #{TimerProbe::RESULT} = "
      get_output.print " #{method_name}_wrapper( "
      get_output.print create_procedure_call_parameters.join(", ")
      get_output.puts " );"
    end

    def copy_array_param_to_ruby(par, param, ruby_param)
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
    hipMemcpy((void *) _boast_n_ary->ptr, #{par}, _boast_array_size, hipMemcpyDeviceToHost);
EOF
      end
      get_output.print <<EOF
    hipFree( (void *) #{par});
  } else if (TYPE(_boast_rb_ptr) == T_STRING) {
EOF
      if param.direction == :out or param.direction == :inout then
        get_output.print <<EOF
    size_t _boast_array_size = RSTRING_LEN(_boast_rb_ptr);
    hipMemcpy((void *) RSTRING_PTR(_boast_rb_ptr), #{par}, _boast_array_size, hipMemcpyDeviceToHost);
EOF
      end
      get_output.print <<EOF
    hipFree( (void *) #{par});
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for %s, expecting array!", "#{param}");
  }
EOF
    end

    def store_results
      store_results_old
      get_output.print "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"duration\")),rb_float_new((double)_boast_duration*(double)1e-9));\n"
    end

  end

end
