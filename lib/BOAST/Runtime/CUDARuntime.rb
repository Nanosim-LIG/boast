module BOAST

  # @private
  module CUDARuntime
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
      get_output.puts "#include <cuda_runtime.h>"
    end

    def fill_library_header
      fill_library_header_old
      get_output.puts "#include <cuda.h>"
    end

    def fill_library_source
      fill_library_source_old
      get_output.write <<EOF
extern "C" {
  #{@procedure.send(:boast_header_s,CUDA)}{
    int _boast_i;
    dim3 dimBlock(_boast_block_size[0], _boast_block_size[1], _boast_block_size[2]);
    dim3 dimGrid(_boast_block_number[0], _boast_block_number[1], _boast_block_number[2]);
    cudaEvent_t __start, __stop;
    float __time;
    cudaEventCreate(&__start);
    cudaEventCreate(&__stop);
    cudaEventRecord(__start, 0);
    for( _boast_i = 0; _boast_i < _boast_repeat; _boast_i ++) {
      #{@procedure.name}<<<dimGrid,dimBlock>>>(#{@procedure.parameters.join(", ")});
    }
    cudaEventRecord(__stop, 0);
    cudaEventSynchronize(__stop);
    cudaEventElapsedTime(&__time, __start, __stop);
    return (unsigned long long int)((double)__time*(double)1e6);
  }
}
EOF
    end

    def copy_array_param_from_ruby(par, param, ruby_param )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      (rb_ptr === ruby_param).pr
      get_output.print <<EOF
  if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    size_t _boast_array_size;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    _boast_array_size = _boast_n_ary->total * na_sizeof[_boast_n_ary->type];
    cudaError_t err = cudaMalloc( (void **) &#{par}, _boast_array_size);
    if (err != cudaSuccess)
      rb_raise(rb_eRuntimeError, "Could not allocate cuda memory for: %s!", "#{param}");
    err = cudaMemcpy(#{par}, (void *) _boast_n_ary->ptr, _boast_array_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      rb_raise(rb_eRuntimeError, "Could not copy memory to device for: %s!", "#{param}");
  } else if (TYPE(_boast_rb_ptr) == T_STRING) {
    size_t _boast_array_size = RSTRING_LEN(_boast_rb_ptr);
    cudaError_t err = cudaMalloc( (void **) &#{par}, _boast_array_size);
    if (err != cudaSuccess)
      rb_raise(rb_eRuntimeError, "Could not allocate cuda memory for: %s!", "#{param}");
    err = cudaMemcpy(#{par}, (void *)RSTRING_PTR(_boast_rb_ptr), _boast_array_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      rb_raise(rb_eRuntimeError, "Could not copy memory to device for: %s!", "#{param}");
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for %s, expecting NArray or String!", "#{param}");
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
    cudaMemcpy((void *) _boast_n_ary->ptr, #{par}, _boast_array_size, cudaMemcpyDeviceToHost);
EOF
      end
      get_output.print <<EOF
    cudaFree( (void *) #{par});
  } else if (TYPE(_boast_rb_ptr) == T_STRING) {
EOF
      if param.direction == :out or param.direction == :inout then
        get_output.print <<EOF
    size_t _boast_array_size = RSTRING_LEN(_boast_rb_ptr);
    cudaMemcpy((void *) RSTRING_PTR(_boast_rb_ptr), #{par}, _boast_array_size, cudaMemcpyDeviceToHost);
EOF
      end
      get_output.print <<EOF
    cudaFree( (void *) #{par});
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
