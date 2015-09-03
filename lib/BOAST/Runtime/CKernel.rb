require 'stringio'
require 'rake'
require 'tempfile'
require 'rbconfig'
require 'systemu'
require 'yaml'
require 'pathname'
require 'os'

module BOAST

  module CompiledRuntime
    def base_name
      return File::split(@marker.path)[1]
    end

    def module_name
      return "Mod_" + base_name.gsub("-","_")
    end

    def directory
      return File::split(@marker.path)[0]
    end

    def module_file_base_name
      return "Mod_" + base_name
    end

    def module_file_base_path
      return directory + "/" + module_file_base_name
    end

    def module_file_path
      return module_file_base_path + RbConfig::CONFIG["DLEXT"]
    end

    def module_file_source
      return module_file_base_path + ".c"
    end

    def module_file_object
      return module_file_base_path + RbConfig::CONFIG["OBJEXT"]
    end

    def library_source
      return directory + "/" + base_name + @@extensions[@lang]
    end

    def library_object
      return directory + "/" + base_name + RbConfig::CONFIG["OBJEXT"]
    end

    def library_path
      return directory + "/" + base_name + RbConfig::CONFIG["DLEXT"]
    end

    def target
      return module_file_path
    end

    def target_depends
      return [ module_file_object, library_object ]
    end

    def create_targets
      file target => target_depends do
        #puts "#{linker} #{ldshared} -o #{target} #{target_depends.join(" ")} #{get_sub_kernels.collect {|f| f.path}).join(" ")} #{ldflags}"
        sh "#{linker} #{ldshared} -o #{target} #{target_depends.join(" ")} #{(@kernel_files.collect {|f| f.path}).join(" ")} #{ldflags}"
      end
      Rake::Task[target].invoke
    end

    def method_name
      return @procedure.name
    end

    def get_sub_kernels
      @kernel_files = []
      @kernels.each { |kernel|
        kernel_file = Tempfile::new([kernel.procedure.name,".#{RbConfig::CONFIG["OBJEXT"]}"])
        kernel.binary.rewind
        kernel_file.write( kernel.binary.read )
        kernel_file.close
        @kernel_files.push(kernel_file)
      }
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
    rb_raise(rb_eArgError, "wrong number of arguments for #{@procedure.name} (%d for #{@procedure.parameters.length})", _boast_argc);
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
      set_decl_module(false)
      get_output.puts "  #{@procedure.properties[:return].type.decl} _boast_ret;" if @procedure.properties[:return]
      get_output.puts "  VALUE _boast_stats = rb_hash_new();"
      get_output.puts "  VALUE _boast_rb_ptr = Qnil;"
    end

    def fill_module_file_source
      fill_module_header
      @probes.map(&:header)
      @procedure.boast_header(@lang)

      fill_module_preamble

      get_output.puts "VALUE method_run(int _boast_argc, VALUE *_boast_argv, VALUE _boast_self) {"
      increment_indent_level

      fill_check_args

      argc = @procedure.parameters.length
      argv = Variable::new("_boast_argv", CustomType, :type_name => "VALUE", :dimension => [ Dimension::new(0,argc-1) ] )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      set_transition("VALUE", "VALUE", :default,  CustomType::new(:type_name => "VALUE"))

      fill_decl_module_params

      @probes.reverse.map(&:decl)

      get_params_value(argv, rb_ptr)

      @probes.map(&:configure)

      @probes.reverse.map(&:start)

      create_procedure_call

      @probes.map(&:stop)

      @probes.map(&:compute)

      get_results(argv, rb_ptr)

      store_result

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
      create_module_file_source
      create_library_source
    end

    def load_module
      require module_file_path
    end

    def target_sources
      return [ module_file_source, library_source ]
    end

    def cleanup
      ([target] + target_depends + target_sources).each { |fn|
        File::unlink(fn)
      }
      @kernel_files.each { |f|
        f.unlink
      }
    end

    def build
      @marker = Tempfile::new([@procedure.name,""])

      get_sub_kernels

      create_sources

      create_targets

      load_module

      cleanup

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

  end

  module CUDARuntime
    include CRuntime

    alias fill_library_source_old fill_library_source
    alias fill_library_header_old fill_library_header
    alias fill_module_header_old fill_module_header

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
    include OpenCLRuntime
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


    @@extensions = {
      C => ".c",
      CUDA => ".cu",
      FORTRAN => ".f90"
    }

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

    def create_module_source(path)
      previous_lang = get_lang
      previous_output = get_output
      set_lang( C )
      extension = @@extensions[@lang]
      extension += "comp" if @architecture == MPPA
      module_file_name = File::split(path.chomp(extension))[0] + "/Mod_" + File::split(path.chomp(extension))[1].gsub("-","_") + ".c"
      module_name = File::split(module_file_name.chomp(File::extname(module_file_name)))[1]
      module_file = File::open(module_file_name,"w+")
      set_output( module_file )
      fill_module(module_file, module_name)
      if debug_source? then
        module_file.rewind
        puts module_file.read
      end
      module_file.close
      set_lang( previous_lang )
      set_output( previous_output )
      return [module_file_name, module_name]
    end

    def save_binary(target)
      f = File::open(target,"rb")
      @binary = StringIO::new
      @binary.write( f.read )
      f.close
    end

    def save_multibinary(target)
      f = File::open(target,"rb")
      @multibinary = StringIO::new
      @multibinary.write( f.read )
      f.close
      @multibinary_path = f.path
    end

    def create_source
      extension = @@extensions[@lang]
      extension += "comp" if @architecture == MPPA
      source_file = Tempfile::new([@procedure.name,extension])
      path = source_file.path
      #target = path.chomp(File::extname(path))+".#{RbConfig::CONFIG["OBJEXT"]}"
      target = path.chomp(extension)
      if @architecture == MPPA then
        target += ".bincomp"
      else
        target += ".#{RbConfig::CONFIG["OBJEXT"]}"
      end
      fill_code(source_file)
      if debug_source? then
        source_file.rewind
        puts source_file.read
      end
      source_file.close
      return [source_file, path, target]
    end

    def create_source_io(origin_source_path)
      extension = @@extensions[@lang]
      extension_comp = extension + "comp"
      extension_io = extension + "io"
      path = origin_source_path.chomp(extension_comp)+@@extensions[@lang]+"io"
      target = origin_source_path.chomp(extension_comp)+".bin"+"io"
      source_file = File::open(path, "w+")
      fill_code(source_file,true)
      if debug_source? then
        source_file.rewind
        puts source_file.read
      end
      source_file.close
      return [source_file, path, target]
    end

    def create_ffi_module(module_name, module_final)
      s =<<EOF
      require 'ffi'
      require 'narray_ffi'
      module #{module_name}
        extend FFI::Library
        ffi_lib "#{module_final}"
        attach_function :#{@procedure.name}#{@lang == FORTRAN ? "_" : ""}, [ #{@procedure.parameters.collect{ |p| ":"+p.decl_ffi.to_s }.join(", ")} ], :#{@procedure.properties[:return] ? @procedure.properties[:return].type.decl_ffi : "void" }
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
              ret = #{@procedure.name}#{@lang == FORTRAN ? "_" : ""}(*t_args)
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

    def create_mppa_target(path)
      extension = @@extensions[@lang] + ".comp"
      multibin = path.chomp(extension) + ".mpk"
      
      return multibin
    end

    def build(options = {})
      compiler_options = BOAST::get_compiler_options
      compiler_options.update(options)
      return build_opencl(compiler_options) if @lang == CL

      linker, ldshared, ldflags = setup_compilers(compiler_options)

      extension = @@extensions[@lang]

      source_file, path, target = create_source

      if @architecture == MPPA then
        source_file_io, path_io, target_io = create_source_io(path)

        multibin = create_mppa_target(path)
        file multibin => [target_io, target] do
          sh "k1-create-multibinary --clusters #{target} --clusters-names \"comp-part\" --boot #{target_io} --bootname \"io-part\" -T #{multibin}"
        end
        Rake::Task[multibin].invoke
        save_multibinary(multibin)
      end

      if not ffi? then
        module_file_name, module_name = create_module_source(path)
        module_target = module_file_name.chomp(File::extname(module_file_name))+"."+RbConfig::CONFIG["OBJEXT"]
        module_final = module_file_name.chomp(File::extname(module_file_name))+"."+RbConfig::CONFIG["DLEXT"]
      else
        module_final = path.chomp(File::extname(path))+"."+RbConfig::CONFIG["DLEXT"]
        module_name = "Mod_" + File::split(path.chomp(File::extname(path)))[1].gsub("-","_")
      end

      kernel_files = get_sub_kernels

      if @architecture == MPPA then
        file module_final => [module_target] do
          sh "#{linker} #{ldshared} -o #{module_final} #{module_target} #{ldflags}"
        end
      elsif not ffi? then
        file module_final => [module_target, target] do
          #puts "#{linker} #{ldshared} -o #{module_final} #{module_target} #{target} #{kernel_files.join(" ")} #{ldflags}"
          sh "#{linker} #{ldshared} -o #{module_final} #{module_target} #{target} #{(kernel_files.collect {|f| f.path}).join(" ")} #{ldflags}"
        end
      else
        file module_final => [target] do
          #puts "#{linker} #{ldshared} -o #{module_final} #{target} #{kernel_files.join(" ")} #{ldflags}"
          sh "#{linker} #{ldshared} -o #{module_final} #{target} #{(kernel_files.collect {|f| f.path}).join(" ")} #{ldflags}"
        end
      end

      Rake::Task[module_final].invoke
      if ffi? then
        create_ffi_module(module_name, module_final)
      else
        require(module_final)
      end

      eval "self.extend(#{module_name})"
      save_binary(target)
      
      if not ffi? then
        [target, module_target, module_file_name, module_final].each { |fn|
          File::unlink(fn)
        }
      else
        [target, module_final].each { |fn|
          File::unlink(fn)
        }
      end
      
      if @architecture == MPPA then
        [target_io].each { |fn|
          File::unlink(fn)
        }
      end

      kernel_files.each { |f|
        f.unlink
      }
      return self
    end

    def fill_code(source_file, io=false)
      if io then
        code = @code_io
      else
        code = @code
      end
 
      code.rewind
      source_file.puts "#include <inttypes.h>" if @lang == C or @lang == CUDA
      source_file.puts "#include <cuda.h>" if @lang == CUDA
      source_file.puts "#include <mppaipc.h>" if @architecture == MPPA
      source_file.puts "#include <mppa/osconfig.h>" if @architecture == MPPA
      # check for too long FORTRAN lines
      if @lang == FORTRAN then
        code.each_line { |line|
          # check for omp pragmas
          if line.match(/^\s*!\$/) then
            if line.match(/^\s*!\$(omp|OMP)/) then
              chunks = line.scan(/.{1,#{FORTRAN_LINE_LENGTH-7}}/)
              source_file.puts chunks.join("&\n!$omp&")
            else
              chunks = line.scan(/.{1,#{FORTRAN_LINE_LENGTH-4}}/)
              source_file.puts chunks.join("&\n!$&")
            end
          elsif line.match(/^\w*!/) then
            source_file.write line
          else
            chunks = line.scan(/.{1,#{FORTRAN_LINE_LENGTH-2}}/)
            source_file.puts chunks.join("&\n&")
          end
        }
      else
        source_file.write code.read
      end
      if @architecture == MPPA then
        source_file.write <<EOF
int main(int argc, const char* argv[]) {
EOF
        if io then #IO Code
          #Parameters declaration
          if @architecture == MPPA then
            @procedure.parameters.each { |param|
              source_file.write "    #{param.type.decl}"
              source_file.write "*" if param.dimension
              source_file.write " #{param.name};\n"
            }
          end
          
          #Cluster list declaration
          source_file.write <<EOF
    uint32_t* _clust_list;
    int _nb_clust;
EOF

          #Receiving parameters from Host
          source_file.write <<EOF
    int _mppa_from_host_size, _mppa_from_host_var, _mppa_to_host_size, _mppa_to_host_var, _mppa_tmp_size, _mppa_pid[16], i;
    _mppa_from_host_size = mppa_open("/mppa/buffer/board0#mppa0#pcie0#2/host#2", O_RDONLY);
    _mppa_from_host_var = mppa_open("/mppa/buffer/board0#mppa0#pcie0#3/host#3", O_RDONLY);
EOF
          @procedure.parameters.each { |param|
            if param.direction == :in or param.direction == :inout then
              if param.dimension then
                source_file.write <<EOF
    mppa_read(_mppa_from_host_size, &_mppa_tmp_size, sizeof(_mppa_tmp_size));
    #{param.name} = malloc(_mppa_tmp_size);
    mppa_read(_mppa_from_host_var, #{param.name}, _mppa_tmp_size);
EOF
              else
                source_file.write <<EOF
    mppa_read(_mppa_from_host_var, &#{param.name}, sizeof(#{param.name}));
EOF
              end
            end
          }

          #Receiving cluster list
          source_file.write <<EOF
    mppa_read(_mppa_from_host_size, &_mppa_tmp_size, sizeof(_mppa_tmp_size));
    _clust_list = malloc(_mppa_tmp_size);
    _nb_clust = _mppa_tmp_size / sizeof(uint32_t);
    mppa_read(_mppa_from_host_var, _clust_list, _mppa_tmp_size);
EOF

          source_file.write <<EOF
    mppa_close(_mppa_from_host_size);
    mppa_close(_mppa_from_host_var);
EOF
          #Spawning cluster
          source_file.write <<EOF
    for(i=0; i<_nb_clust;i++){
        _mppa_pid[i] = mppa_spawn(_clust_list[i], NULL, "comp-part", NULL, NULL);
    }
EOF
          source_file.write "    #{@procedure.name}("
          @procedure.parameters.each_with_index { |param, i|
            source_file.write ", " unless i == 0
            if !param.dimension then
              if param.direction == :out or param.direction == :inout then
                source_file.write "&"
              end
            end
            source_file.write param.name
          }
          source_file.write ");\n"
        else #Compute code
          source_file.write "    #{@procedure.name}();\n"
        end
        
        
        #Sending results to Host
        if io then #IO Code
          source_file.write <<EOF
    for(i=0; i< _nb_clust; i++){
        mppa_waitpid(_mppa_pid[i], NULL, 0);
    }
    _mppa_to_host_size = mppa_open("/mppa/buffer/host#4/board0#mppa0#pcie0#4", O_WRONLY);
    _mppa_to_host_var = mppa_open("/mppa/buffer/host#5/board0#mppa0#pcie0#5", O_WRONLY);
EOF
          @procedure.parameters.each { |param| 
            if param.direction == :out or param.direction == :inout then
              if param.dimension then
                source_file.write <<EOF
    _mppa_tmp_size = #{param.dimension.size};
    mppa_write(_mppa_to_host_size, &_mppa_tmp_size, sizeof(_mppa_tmp_size));
    mppa_write(_mppa_to_host_var, #{param.name}, _mppa_tmp_size);
EOF
              else
                source_file.write <<EOF
    mppa_write(_mppa_to_host_var, &#{param.name}, sizeof(#{param.name}));
EOF
              end
            end
          }
          source_file.write <<EOF
    mppa_close(_mppa_to_host_size);
    mppa_close(_mppa_to_host_var);
EOF
        else #Compute code
        end
        source_file.write <<EOF
    mppa_exit(0);
    return 0;
}
EOF
      elsif @lang == CUDA then
        source_file.write <<EOF
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
      code.rewind
    end

    def module_header(module_file)
      module_file.print <<EOF
#include "ruby.h"
#include <inttypes.h>
#ifdef HAVE_NARRAY_H
#include "narray.h"
#endif
EOF


      if @lang == CUDA then
        module_file.print "#include <cuda_runtime.h>\n"
      end
      if @architecture == MPPA then
        module_file.print "#include <mppaipc.h>\n"
        module_file.print "#include <mppa_mon.h>\n"
      end
    end

    def module_preamble(module_file, module_name)
      module_file.print <<EOF
VALUE #{module_name} = Qnil;
void Init_#{module_name}();
VALUE method_run(int _boast_argc, VALUE *_boast_argv, VALUE _boast_self);
void Init_#{module_name}() {
  #{module_name} = rb_define_module("#{module_name}");
  rb_define_method(#{module_name}, "run", method_run, -1);
}
EOF
    end

    def check_args(module_file)
      module_file.print <<EOF
  VALUE _boast_rb_opts;
  if( _boast_argc < #{@procedure.parameters.length} || _boast_argc > #{@procedure.parameters.length + 1} )
    rb_raise(rb_eArgError, "wrong number of arguments for #{@procedure.name} (%d for #{@procedure.parameters.length})", _boast_argc);
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

    def get_params_value(module_file, argv, rb_ptr)
      set_decl_module(true)
      @procedure.parameters.each_index do |i|
        param = @procedure.parameters[i]
        if not param.dimension then
          case param.type
          when Int 
            (param === FuncCall::new("NUM2INT", argv[i])).pr if param.type.size == 4
            (param === FuncCall::new("NUM2LONG", argv[i])).pr if param.type.size == 8
          when Real
            (param === FuncCall::new("NUM2DBL", argv[i])).pr
          end
        else
          (rb_ptr === argv[i]).pr
          if @lang == CUDA then
            module_file.print <<EOF
  if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    size_t _boast_array_size;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    _boast_array_size = _boast_n_ary->total * na_sizeof[_boast_n_ary->type];
    cudaMalloc( (void **) &#{param.name}, _boast_array_size);
    cudaMemcpy(#{param.name}, (void *) _boast_n_ary->ptr, _boast_array_size, cudaMemcpyHostToDevice);
  } else {
    rb_raise(rb_eArgError, "wrong type of argument %d", #{i});
  }
EOF
          else
            module_file.print <<EOF
  if (TYPE(_boast_rb_ptr) == T_STRING) {
    #{param.name} = (void *) RSTRING_PTR(_boast_rb_ptr);
  } else if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    #{param.name} = (void *) _boast_n_ary->ptr;
  } else {
    rb_raise(rb_eArgError, "wrong type of argument %d", #{i});
  }
EOF
          end
        end
      end
      set_decl_module(false)
    end

    def decl_module_params(module_file)
      set_decl_module(true)
      @procedure.parameters.each { |param| 
        param_copy = param.copy
        param_copy.constant = nil
        param_copy.direction = nil
        param_copy.decl
      }
      set_decl_module(false)
      module_file.print "  #{@procedure.properties[:return].type.decl} _boast_ret;\n" if @procedure.properties[:return]
      module_file.print "  VALUE _boast_stats = rb_hash_new();\n"

    end

    def get_cuda_launch_bounds(module_file)
      module_file.print <<EOF
  size_t _boast_block_size[3] = {1,1,1};
  size_t _boast_block_number[3] = {1,1,1};
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

    def create_procedure_call(module_file)
      if @architecture == MPPA then
        mppa_load_id = Variable::new("_mppa_load_id", Int)
        mppa_pid = Variable::new("_mppa_pid", Int)
        mppa_fd_size = Variable::new("_mppa_fd_size", Int)
        fd = Variable::new("_mppa_fd_var", Int)
        size = Variable::new("_mppa_size", Int)
        avg_pwr = Variable::new("avg_pwr", Real, :size => 4)
        energy = Variable::new("energy", Real, :size => 4)
        mppa_duration = Variable::new("mppa_duration", Real, :size => 4)
        mppa_clust_list_size = Variable::new("_mppa_clust_list_size", Int)
        mppa_clust_nb = Variable::new("_mppa_clust_nb", Int, :size => 4)
        mppa_load_id.decl
        mppa_pid.decl
        mppa_fd_size.decl
        fd.decl
        size.decl
        avg_pwr.decl
        energy.decl
        mppa_duration.decl
        mppa_clust_list_size.decl
        mppa_clust_nb.decl
        module_file.print <<EOF
  uint32_t* _mppa_clust_list;
  mppa_mon_ctx_t* mppa_ctx;
  mppa_mon_sensor_t pwr_sensor[] = {MPPA_MON_PWR_MPPA0};
  mppa_mon_measure_report_t* mppa_report;
  mppa_mon_open(0, &mppa_ctx);
  mppa_mon_measure_set_sensors(mppa_ctx, pwr_sensor, 1);
EOF
        module_file.print "  _mppa_load_id = mppa_load(0, 0, 0, \"#{@multibinary_path}\");\n"
        module_file.print "  mppa_mon_measure_start(mppa_ctx);\n"
        module_file.print "  _mppa_pid = mppa_spawn(_mppa_load_id, NULL, \"io-part\", NULL, NULL);\n"
        module_file.print "  _mppa_fd_size = mppa_open(\"/mppa/buffer/board0#mppa0#pcie0#2/host#2\", O_WRONLY);\n"
        module_file.print "  _mppa_fd_var = mppa_open(\"/mppa/buffer/board0#mppa0#pcie0#3/host#3\", O_WRONLY);\n"
        

        # Sending parameters to IO Cluster
        @procedure.parameters.each { |param|
          if param.direction == :in or param.direction == :inout then
            if param.dimension then
              size === param.dimension.size
              module_file.print "  mppa_write(_mppa_fd_size, &_mppa_size, sizeof(_mppa_size));\n"
              module_file.print "  mppa_write(_mppa_fd_var, #{param.name}, sizeof(#{param.name}));\n"
            else
              module_file.print "  mppa_write(_mppa_fd_var, &#{param.name}, sizeof(#{param.name}));\n"
            end
          end
        }

        # Sending cluster list
        module_file.print <<EOF
  if(_boast_rb_opts != Qnil){
    _boast_rb_ptr = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("clust_list")));
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
  
  _mppa_clust_list_size = sizeof(uint32_t)*_mppa_clust_nb;
  mppa_write(_mppa_fd_size, &_mppa_clust_list_size, sizeof(_mppa_clust_list_size));
  mppa_write(_mppa_fd_var, _mppa_clust_list, _mppa_clust_list_size);
  free(_mppa_clust_list);
EOF

        module_file.print "  mppa_close(_mppa_fd_var);\n"
        module_file.print "  mppa_close(_mppa_fd_size);\n"

        module_file.print "  _mppa_fd_size = mppa_open(\"/mppa/buffer/host#4/board0#mppa0#pcie0#4\", O_RDONLY);\n"
        module_file.print "  _mppa_fd_var = mppa_open(\"/mppa/buffer/host#5/board0#mppa0#pcie0#5\", O_RDONLY);\n"
        # Receiving parameters
        @procedure.parameters.each { |param|
          if param.direction == :out or param.direction == :inout then
            if param.dimension then
              module_file.print "  mppa_read(_mppa_fd_size, &_mppa_size, sizeof(_mppa_size));\n"
              module_file.print "  mppa_read(_mppa_fd_var, #{param.name}, _mppa_size);\n"
            else
              module_file.print "  mppa_read(_mppa_fd_var, &#{param.name}, sizeof(#{param.name}));\n"
            end
          end
        }
        module_file.print "  mppa_close(_mppa_fd_var);\n"
        module_file.print "  mppa_close(_mppa_fd_size);\n"

        # TODO : Retrieving timers

        module_file.print "  mppa_waitpid(_mppa_pid, NULL, 0);\n"
        module_file.print "  mppa_mon_measure_stop(mppa_ctx, &mppa_report);\n"
        module_file.print "  mppa_unload(_mppa_load_id);\n"
      else
        if @lang == CUDA then
          module_file.print "  #{TimerProbe::RESULT} = "
        elsif @procedure.properties[:return] then
          module_file.print "  _boast_ret = "
        end
        module_file.print "  #{@procedure.name}"
        module_file.print "_" if @lang == FORTRAN
        module_file.print "_wrapper" if @lang == CUDA
        module_file.print "("
        params = []
        if(@lang == FORTRAN) then
          @procedure.parameters.each { |param|
            if param.dimension then
              params.push( param.name )
            else
              params.push( "&"+param.name )
            end
          }
        else 
          @procedure.parameters.each { |param|
            if param.dimension then
              params.push( param.name )
            elsif param.direction == :out or param.direction == :inout then
              params.push( "&"+param.name )
            else
              params.push( param.name )
            end
          }
        end
        if @lang == CUDA then
          params.push( "_boast_block_number", "_boast_block_size" )
        end
        module_file.print params.join(", ")
        module_file.print "  );\n"
      end
    end

    def get_results(module_file, argv, rb_ptr)
      set_decl_module(true)
      if @lang == CUDA then
        @procedure.parameters.each_index do |i|
          param = @procedure.parameters[i]
          if param.dimension then
            (rb_ptr === argv[i]).pr
            module_file.print <<EOF
  if ( IsNArray(_boast_rb_ptr) ) {
EOF
            if param.direction == :out or param.direction == :inout then
            module_file.print <<EOF
    struct NARRAY *_boast_n_ary;
    size_t _boast_array_size;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    _boast_array_size = _boast_n_ary->total * na_sizeof[_boast_n_ary->type];
    cudaMemcpy((void *) _boast_n_ary->ptr, #{param.name}, _boast_array_size, cudaMemcpyDeviceToHost);
EOF
            end
            module_file.print <<EOF
    cudaFree( (void *) #{param.name});
  } else {
    rb_raise(rb_eArgError, "wrong type of argument %d", #{i});
  }
EOF
          end
        end
      else
        first = true
        @procedure.parameters.each_with_index do |param,i|
          if param.scalar_output? then
            if first then
              module_file.print "  VALUE _boast_refs = rb_hash_new();\n"
              module_file.print "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"reference_return\")),_boast_refs);\n"
              first = false
            end
            case param.type
            when Int
              module_file.print "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_int_new((long long)#{param}));\n" if param.type.signed?
              module_file.print "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_int_new((unsigned long long)#{param}));\n" if not param.type.signed?
            when Real
              module_file.print "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_float_new((double)#{param}));\n"
            end
          end
        end
      end
      set_decl_module(false)
    end

    def store_result(module_file)
      if @procedure.properties[:return] then
        type_ret = @procedure.properties[:return].type
        module_file.print "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_int_new((long long)_boast_ret));\n" if type_ret.kind_of?(Int) and type_ret.signed
        module_file.print "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_int_new((unsigned long long)_boast_ret));\n" if type_ret.kind_of?(Int) and not type_ret.signed
        module_file.print "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_float_new((double)_boast_ret));\n" if type_ret.kind_of?(Real)
      end
      if @architecture == MPPA then
        module_file.print <<EOF
  rb_hash_aset(_boast_stats,ID2SYM(rb_intern("avg_pwr")),rb_float_new(avg_pwr));
  rb_hash_aset(_boast_stats,ID2SYM(rb_intern("energy")),rb_float_new(energy));
  rb_hash_aset(_boast_stats,ID2SYM(rb_intern("mppa_duration")), rb_float_new(mppa_duration));
EOF
      end
    end

    def fill_module(module_file, module_name)
      module_header(module_file)
      @probes.map(&:header)
      @procedure.boast_header(@lang)

      module_preamble(module_file, module_name)

      module_file.puts "VALUE method_run(int _boast_argc, VALUE *_boast_argv, VALUE _boast_self) {"
      increment_indent_level
      check_args(module_file)

      argc = @procedure.parameters.length
      argv = Variable::new("_boast_argv", CustomType, :type_name => "VALUE", :dimension => [ Dimension::new(0,argc-1) ] )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      set_transition("VALUE", "VALUE", :default,  CustomType::new(:type_name => "VALUE"))
      rb_ptr.decl

      decl_module_params(module_file)
      @probes.reverse.map(&:decl)

      get_params_value(module_file, argv, rb_ptr)

      if @lang == CUDA then
        module_file.print get_cuda_launch_bounds(module_file)
      end

      @probes.map(&:configure)

      @probes.reverse.map(&:start)

      create_procedure_call(module_file)

      @probes.map(&:stop)

      if @architecture == MPPA then
        module_file.print <<EOF
  avg_pwr = 0;
  energy = 0;
  int i;
  for(i=0; i < mppa_report->count; i++){
    avg_pwr += mppa_report->measures[i].avg_power;
    energy += mppa_report->measures[i].total_energy;
  } 
  avg_pwr = avg_pwr/(float) mppa_report->count;
  mppa_duration = mppa_report->total_time;
  mppa_mon_measure_free_report(mppa_report);
  mppa_mon_close(mppa_ctx);
EOF
      end

      @probes.map(&:compute)

      get_results(module_file, argv, rb_ptr)

      store_result(module_file)

      module_file.print "  return _boast_stats;\n"
      decrement_indent_level
      module_file.print "}"
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
