require 'stringio'
require 'rake'
require 'tempfile'
require 'rbconfig'
require 'systemu'
require 'yaml'
require 'pathname'
require 'os'

class Dir
  module Tmpname
    module_function
    def make_tmpname(prefix_suffix, n)
      case prefix_suffix
      when String
        prefix = prefix_suffix
        suffix = ""
      when Array
        prefix = prefix_suffix[0]
        suffix = prefix_suffix[1]
      else
        raise ArgumentError, "unexpected prefix_suffix: #{prefix_suffix.inspect}"
      end
      t = Time.now.strftime("%Y%m%d")
      path = "#{prefix}#{t}_#{$$}_#{rand(0x100000000).to_s(36)}"
      path << "_#{n}" if n
      path << suffix
    end
  end
end

module BOAST

  module CompiledRuntime
    @@extensions = {
      C => ".c",
      CUDA => ".cu",
      FORTRAN => ".f90"
    }

    def base_name
      return File::split(@marker.path)[1]
    end

    def module_name
      return "Mod_" + base_name#.gsub("-","_")
    end

    def directory
      return File::split(@marker.path)[0]
    end

    def module_file_base_name
      return "Mod_" + base_name
    end

    def module_file_base_path
      return "#{directory}/#{module_file_base_name}"
    end

    def module_file_path
      return "#{module_file_base_path}.#{RbConfig::CONFIG["DLEXT"]}"
    end

    def module_file_source
      return module_file_base_path + ".c"
    end

    def module_file_object
      return "#{module_file_base_path}.#{RbConfig::CONFIG["OBJEXT"]}"
    end

    def library_source
      return directory + "/" + base_name + @@extensions[@lang]
    end

    def library_object
      return "#{directory}/#{base_name}.#{RbConfig::CONFIG["OBJEXT"]}"
    end

    def library_path
      return "#{directory}/#{base_name}.#{RbConfig::CONFIG["DLEXT"]}"
    end

    def target
      return module_file_path
    end

    def target_depends
      return [ module_file_object, library_object ]
    end

    def create_targets( linker, ldshared, ldflags, kernel_files)
      file target => target_depends do
        #puts "#{linker} #{ldshared} -o #{target} #{target_depends.join(" ")} #{(kernel_files.collect {|f| f.path}).join(" ")} #{ldflags}"
        sh "#{linker} #{ldshared} -o #{target} #{target_depends.join(" ")} #{(kernel_files.collect {|f| f.path}).join(" ")} #{ldflags}"
      end
      Rake::Task[target].invoke
    end

    def method_name
      return @procedure.name
    end

    def get_sub_kernels
      kernel_files = []
      @kernels.each { |kernel|
        kernel_file = Tempfile::new([kernel.procedure.name,".#{RbConfig::CONFIG["OBJEXT"]}"])
        kernel.binary.rewind
        kernel_file.write( kernel.binary.read )
        kernel_file.close
        kernel_files.push(kernel_file)
      }
      return kernel_files
    end

    def create_library_source
      f = File::open(library_source,"w+")
      previous_lang = get_lang
      previous_output = get_output
      set_output(f)
      set_lang(@lang)

      fill_library_source

      if debug_source? then
        f.rewind
        puts f.read
      end
      set_output(previous_output)
      set_lang(previous_lang)
      f.close
    end

    def fill_module_header
      get_output.print <<EOF
#include "ruby.h"
#include <inttypes.h>
#ifdef HAVE_NARRAY_H
#include "narray.h"
#endif
EOF
    end

    def fill_module_preamble
      get_output.print <<EOF
VALUE #{module_name} = Qnil;
void Init_#{module_name}();
VALUE method_run(int _boast_argc, VALUE *_boast_argv, VALUE _boast_self);
void Init_#{module_name}() {
  #{module_name} = rb_define_module("#{module_name}");
  rb_define_method(#{module_name}, "run", method_run, -1);
}
EOF
    end

    def fill_check_args
      get_output.print <<EOF
  VALUE _boast_rb_opts;
  if( _boast_argc < #{@procedure.parameters.length} || _boast_argc > #{@procedure.parameters.length + 1} )
    rb_raise(rb_eArgError, "Wrong number of arguments for #{@procedure.name} (%d for #{@procedure.parameters.length})!", _boast_argc);
  _boast_rb_opts = Qnil;
  if( _boast_argc == #{@procedure.parameters.length + 1} ) {
    _boast_rb_opts = _boast_argv[_boast_argc -1];
    if ( _boast_rb_opts != Qnil ) {
      if (TYPE(_boast_rb_opts) != T_HASH)
        rb_raise(rb_eArgError, "Options should be passed as a hash");
    }
  }
EOF
    end

    def fill_decl_module_params
      set_decl_module(true)
      @procedure.parameters.each { |param|
        param_copy = param.copy
        param_copy.constant = nil
        param_copy.direction = nil
        param_copy.decl
      }
      get_output.puts "  #{@procedure.properties[:return].type.decl} _boast_ret;" if @procedure.properties[:return]
      get_output.puts "  VALUE _boast_stats = rb_hash_new();"
      get_output.puts "  VALUE _boast_rb_ptr = Qnil;"
      refs = false
      @procedure.parameters.each_with_index do |param,i|
        refs = true if param.scalar_output?
      end
      if refs then
        get_output.puts "  VALUE _boast_refs = rb_hash_new();"
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"reference_return\")),_boast_refs);"
      end
      set_decl_module(false)
    end

    def copy_scalar_param_from_ruby( param, ruby_param )
      case param.type
      when Int
        (param === FuncCall::new("NUM2INT", ruby_param)).pr if param.type.size == 4
        (param === FuncCall::new("NUM2LONG", ruby_param)).pr if param.type.size == 8
      when Real
        (param === FuncCall::new("NUM2DBL", ruby_param)).pr
      end
    end

    def copy_array_param_from_ruby( param, ruby_param )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      (rb_ptr === ruby_param).pr
      get_output.print <<EOF
  if (TYPE(_boast_rb_ptr) == T_STRING) {
    #{param} = (void *) RSTRING_PTR(_boast_rb_ptr);
  } else if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    #{param} = (void *) _boast_n_ary->ptr;
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for %s, expecting array!", "#{param}");
  }
EOF
    end

    def get_params_value
      argc = @procedure.parameters.length
      argv = Variable::new("_boast_argv", CustomType, :type_name => "VALUE", :dimension => [ Dimension::new(0,argc-1) ] )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      set_decl_module(true)
      @procedure.parameters.each_index do |i|
        param = @procedure.parameters[i]
        if not param.dimension then
          copy_scalar_param_from_ruby(param, argv[i])
        else
          copy_array_param_from_ruby(param, argv[i])
        end
      end
      set_decl_module(false)
    end

    def create_procedure_call
      get_output.print "  _boast_ret = " if @procedure.properties[:return]
      get_output.print " #{method_name}( "
      get_output.print create_procedure_call_parameters.join(", ")
      get_output.puts " );"
    end

    def copy_scalar_param_to_ruby(param, ruby_param)
      if param.scalar_output? then
        case param.type
        when Int
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_int_new((long long)#{param}));" if param.type.signed?
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_int_new((unsigned long long)#{param}));" if not param.type.signed?
        when Real
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_float_new((double)#{param}));"
        end
      end
    end

    def copy_array_param_to_ruby(param, ruby_param)
    end

    def get_results
      argc = @procedure.parameters.length
      argv = Variable::new("_boast_argv", CustomType, :type_name => "VALUE", :dimension => [ Dimension::new(0,argc-1) ] )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      set_decl_module(true)
      @procedure.parameters.each_index do |i|
        param = @procedure.parameters[i]
        if not param.dimension then
          copy_scalar_param_to_ruby(param, argv[i])
        else
          copy_array_param_to_ruby(param, argv[i])
        end
      end
      set_decl_module(false)
    end

    def store_results
      if @procedure.properties[:return] then
        type_ret = @procedure.properties[:return].type
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_int_new((long long)_boast_ret));" if type_ret.kind_of?(Int) and type_ret.signed
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_int_new((unsigned long long)_boast_ret));" if type_ret.kind_of?(Int) and not type_ret.signed
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_float_new((double)_boast_ret));" if type_ret.kind_of?(Real)
      end
    end

    def fill_module_file_source
      fill_module_header
      @probes.map(&:header)
      @procedure.boast_header(@lang)

      fill_module_preamble

      set_transition("VALUE", "VALUE", :default,  CustomType::new(:type_name => "VALUE"))
      get_output.puts "VALUE method_run(int _boast_argc, VALUE *_boast_argv, VALUE _boast_self) {"
      increment_indent_level

      fill_check_args

      fill_decl_module_params

      @probes.reverse.map(&:decl)

      get_params_value

      @probes.map(&:configure)

      @probes.reverse.map(&:start)

      create_procedure_call

      @probes.map(&:stop)

      @probes.map(&:compute)

      get_results

      store_results

      get_output.puts "  return _boast_stats;"
      decrement_indent_level
      get_output.puts "}"
    end

    def create_module_file_source
      f = File::open(module_file_source, "w+")
      previous_lang = get_lang
      previous_output = get_output
      set_output(f)
      set_lang(C)

      fill_module_file_source

      if debug_source? then
        f.rewind
        puts f.read
      end
      set_output(previous_output)
      set_lang(previous_lang)
      f.close
    end

    def create_sources
      create_library_source
      create_module_file_source
    end

    def load_module
      require module_file_path
    end

    def target_sources
      return [ module_file_source, library_source ]
    end

    def cleanup(kernel_files)
      ([target] + target_depends + target_sources).each { |fn|
        File::unlink(fn)
      }
      kernel_files.each { |f|
        f.unlink
      }
    end

    def build(options={})
      compiler_options = BOAST::get_compiler_options
      compiler_options.update(options)
      linker, ldshared, ldflags = setup_compilers(compiler_options)

      @marker = Tempfile::new([@procedure.name,""])

      kernel_files = get_sub_kernels

      create_sources

      create_targets(linker, ldshared, ldflags, kernel_files)

      load_module

      cleanup(kernel_files)

      eval "self.extend(#{module_name})"

      return self
    end
  end

  module CRuntime
    include CompiledRuntime

    def fill_library_header
      get_output.puts "#include <inttypes.h>"
    end

    def fill_library_source
      fill_library_header
      @code.rewind
      get_output.write code.read
    end

    def create_procedure_call_parameters
      params = []
      @procedure.parameters.each { |param|
        if param.dimension then
          params.push( param.name )
        elsif param.direction == :out or param.direction == :inout then
          params.push( "&"+param.name )
        else
          params.push( param.name )
        end
      }
      return params
    end

  end

  module CUDARuntime
    include CRuntime

    alias fill_library_source_old fill_library_source
    alias fill_library_header_old fill_library_header
    alias fill_module_header_old fill_module_header
    alias get_params_value_old get_params_value
    alias fill_decl_module_params_old fill_decl_module_params
    alias create_procedure_call_parameters_old create_procedure_call_parameters

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
  #{@procedure.boast_header_s(CUDA)}{
    dim3 dimBlock(block_size[0], block_size[1], block_size[2]);
    dim3 dimGrid(block_number[0], block_number[1], block_number[2]);
    cudaEvent_t __start, __stop;
    float __time;
    cudaEventCreate(&__start);
    cudaEventCreate(&__stop);
    cudaEventRecord(__start, 0);
    #{@procedure.name}<<<dimGrid,dimBlock>>>(#{@procedure.parameters.join(", ")});
    cudaEventRecord(__stop, 0);
    cudaEventSynchronize(__stop);
    cudaEventElapsedTime(&__time, __start, __stop);
    return (unsigned long long int)((double)__time*(double)1e6);
  }
}
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
    cudaMalloc( (void **) &#{param}, _boast_array_size);
    cudaMemcpy(#{param}, (void *) _boast_n_ary->ptr, _boast_array_size, cudaMemcpyHostToDevice);
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for %s, expecting array!", "#{param}");
  }
EOF
    end

    def fill_decl_module_params
      fill_decl_module_params_old
      get_output.print <<EOF
  size_t _boast_block_size[3] = {1,1,1};
  size_t _boast_block_number[3] = {1,1,1};
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

    def create_procedure_call_parameters
      return create_procedure_call_parameters_old + ["_boast_block_number", "_boast_block_size"]
    end

    def create_procedure_call
      get_output.print "  #{TimerProbe::RESULT} = "
      get_output.print " #{method_name}_wrapper( "
      get_output.print create_procedure_call_parameters.join(", ")
      get_output.puts " );"
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
    cudaMemcpy((void *) _boast_n_ary->ptr, #{param}, _boast_array_size, cudaMemcpyDeviceToHost);
EOF
      end
      get_output.print <<EOF
    cudaFree( (void *) #{param});
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for %s, expecting array!", "#{param}");
  }
EOF
    end

  end

  module FORTRANRuntime
    include CompiledRuntime

    def method_name
      return @procedure.name + "_"
    end

    def fill_library_source
      @code.rewind
      @code.each_line { |line|
        # check for omp pragmas
        if line.match(/^\s*!\$/) then
          if line.match(/^\s*!\$(omp|OMP)/) then
            chunks = line.scan(/.{1,#{FORTRAN_LINE_LENGTH-7}}/)
            get_output.puts chunks.join("&\n!$omp&")
          else
            chunks = line.scan(/.{1,#{FORTRAN_LINE_LENGTH-4}}/)
            get_output.puts chunks.join("&\n!$&")
          end
        elsif line.match(/^\w*!/) then
          get_output.write line
        else
          chunks = line.scan(/.{1,#{FORTRAN_LINE_LENGTH-2}}/)
          get_output.puts chunks.join("&\n&")
        end
      }
    end

    def create_procedure_call_parameters
      params = []
      @procedure.parameters.each { |param|
        if param.dimension then
          params.push( param.name )
        else
          params.push( "&"+param.name )
        end
      }
      return params
    end

  end

  module FFIRuntime
    def init
      if @lang = FORTRAN then
        extend FORTRANRuntime
      elsif @lang = C then
        extend CRuntime
      else
        raise "FFI only supports C or FORTRAN!"
      end
    end

    def target
      return library_path
    end

    def target_depends
      return [ library_object ]
    end

    def target_sources
      return [ library_source ]
    end

    def load_module
      create_ffi_module
    end

    def create_sources
      create_library_source
    end

    def create_ffi_module
      s =<<EOF
      require 'ffi'
      require 'narray_ffi'
      module #{module_name}
        extend FFI::Library
        ffi_lib "#{library_path}"
        attach_function :#{method_name}, [ #{@procedure.parameters.collect{ |p| ":"+p.decl_ffi.to_s }.join(", ")} ], :#{@procedure.properties[:return] ? @procedure.properties[:return].type.decl_ffi : "void" }
        def run(*args)
          if args.length < @procedure.parameters.length or args.length > @procedure.parameters.length + 1 then
            raise "Wrong number of arguments for \#{@procedure.name} (\#{args.length} for \#{@procedure.parameters.length})"
          else
            ev_set = nil
            if args.length == @procedure.parameters.length + 1 then
              options = args.last
              if options[:PAPI] then
                require 'PAPI'
                ev_set = PAPI::EventSet::new
                ev_set.add_named(options[:PAPI])
              end
            end
            t_args = []
            r_args = {}
            if @lang == FORTRAN then
              @procedure.parameters.each_with_index { |p, i|
                if p.decl_ffi(true) != :pointer then
                  arg_p = FFI::MemoryPointer::new(p.decl_ffi(true))
                  arg_p.send("write_\#{p.decl_ffi(true)}",args[i])
                  t_args.push(arg_p)
                  r_args[p] = arg_p if p.scalar_output?
                else
                  t_args.push( args[i] )
                end
              }
            else
              @procedure.parameters.each_with_index { |p, i|
                if p.scalar_output? then
                  arg_p = FFI::MemoryPointer::new(p.decl_ffi(true))
                  arg_p.send("write_\#{p.decl_ffi(true)}",args[i])
                  t_args.push(arg_p)
                  r_args[p] = arg_p
                else
                  t_args.push( args[i] )
                end
              }
            end
            results = {}
            counters = nil
            ev_set.start if ev_set
            begin
              start = Time::new
              ret = #{method_name}(*t_args)
              stop = Time::new
            ensure
              if ev_set then
                counters = ev_set.stop
                ev_set.cleanup
                ev_set.destroy
              end
            end
            results = { :start => start, :stop => stop, :duration => stop - start, :return => ret }
            results[:PAPI] = Hash[[options[:PAPI]].flatten.zip(counters)] if ev_set
            if r_args.length > 0 then
              ref_return = {}
              r_args.each { |p, p_arg|
                ref_return[p.name.to_sym] = p_arg.send("read_\#{p.decl_ffi(true)}")
              }
              results[:reference_return] = ref_return
            end
            return results
          end
        end
      end
EOF
      eval s
    end

  end

  class CKernel
    include Compilers
    include Rake::DSL
    include Inspectable
    include PrivateStateAccessor
    include TypeTransition

    attr_accessor :code
    attr_accessor :procedure
    attr_accessor :lang
    attr_accessor :binary
    attr_accessor :kernels
    attr_accessor :cost_function
    
    def initialize(options={})
      if options[:code] then
        @code = options[:code]
      elsif get_chain_code
        @code = get_output
        @code.seek(0,SEEK_END)
      else
        @code = StringIO::new
      end
      set_output(@code)
      if options[:kernels] then
        @kernels = options[:kernels]
      else
        @kernels  = []
      end
      if options[:lang] then
        @lang = options[:lang]
      else
        @lang = get_lang
      end
      if options[:architecture] then
        @architecture = options[:architecture]
      else
        @architecture = get_architecture
      end
      @probes = [TimerProbe, PAPIProbe]

      case @lang
      when CL
        extend OpenCLRuntime
      when CUDA
        extend CUDARuntime
      when FORTRAN
        if ffi? then
          extend FFIRuntime
        else
          extend FORTRANRuntime
        end
      else
        if ffi? then
          extend FFIRuntime
        else
          extend CRuntime
        end
      end
    end

    def set_io
      @code_io = StringIO::new unless @code_io
      set_output(@code_io)
    end

    def set_comp
      set_output(@code)
    end

    def print
      @code.rewind
      puts @code.read
    end

    def to_s
      @code.rewind
      return code.read
    end

    def method_missing(meth, *args, &block)
     if meth.to_s == "run" then
       build
       run(*args,&block)
     else
       super
     end
    end

    def cost(*args)
      @cost_function.call(*args)
    end
  end
end
