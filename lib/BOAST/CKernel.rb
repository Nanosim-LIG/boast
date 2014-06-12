require 'stringio'
require 'rubygems'
require 'rake'
require 'tempfile'
require 'rbconfig'
require 'systemu'
require 'yaml'

module BOAST
  @@verbose = false
  @@compiler_default_options = {
    :FC => 'gfortran',
    :FCFLAGS => '-O2 -Wall',
    :CC => 'gcc',
    :CFLAGS => '-O2 -Wall',
    :CXX => 'g++',
    :CXXFLAGS => '-O2 -Wall',
    :NVCC => 'nvcc',
    :NVCCFLAGS => '-O2',
    :LDFLAGS => '',
    :CLFLAGS => '',
    :openmp => false
  }
  
  @@openmp_default_flags = {
    "gcc" => "-fopenmp",
    "icc" => "-openmp",
    "gfortran" => "-fopenmp",
    "ifort" => "-openmp",
    "g++" => "-fopenmp",
    "icpc" => "-openmp"
  }

  def BOAST::read_boast_config
    home_config_dir = ENV["XDG_CONFIG_HOME"]
    home_config_dir = "#{Dir.home}/.config" if not home_config_dir
    Dir.mkdir( home_config_dir ) if not File::exist?( home_config_dir )
    return if not File::directory?(home_config_dir)
    boast_config_dir = "#{home_config_dir}/BOAST"
    Dir.mkdir( boast_config_dir ) if not File::exist?( boast_config_dir )
    compiler_options_file = "#{boast_config_dir}/compiler_options"
    if File::exist?( compiler_options_file ) then
      File::open( compiler_options_file, "r" ) { |f|
        @@compiler_default_options.update( YAML::load( f.read ) )
      }
    else
      File::open( compiler_options_file, "w" ) { |f|
        f.write YAML::dump( @@compiler_default_options )
      }
    end
    openmp_flags_file = "#{boast_config_dir}/openmp_flags"
    if File::exist?( openmp_flags_file ) then
      File::open( openmp_flags_file, "r" ) { |f|
        @@openmp_default_flags.update( YAML::load( f.read ) )
      }
    else
      File::open( openmp_flags_file, "w" ) { |f|
        f.write YAML::dump( @@openmp_default_flags )
      }
    end
    @@compiler_default_options.each_key { |k|
      @@compiler_default_options[k] = ENV[k.to_s] if ENV[k.to_s]
    }
    @@compiler_default_options[:LD] = ENV["LD"] if ENV["LD"]
    @@verbose = ENV["VERBOSE"] if ENV["VERBOSE"]
  end

  BOAST::read_boast_config

  def BOAST::get_openmp_flags
    return @@openmp_default_flags.clone
  end

  def BOAST::get_compiler_options
    return @@compiler_default_options.clone
  end


  def BOAST::get_verbose
    return @@verbose
  end

  def BOAST::set_verbose(verbose)
    @@verbose = verbose
  end

  class CKernel
    include Rake::DSL
    attr_accessor :code
    attr_accessor :procedure
    attr_accessor :lang
    attr_accessor :binary
    attr_accessor :kernels
    
    def initialize(options={})
      if options[:code] then
        @code = options[:code]
      elsif BOAST::get_chain_code
        @code = BOAST::get_output
        @code.seek(0,SEEK_END)
      else
        @code = StringIO::new
      end
      BOAST::set_output( @code )
      if options[:kernels] then
        @kernels = options[:kernels]
      else
        @kernels  = []
      end
      if options[:lang] then
        @lang = options[:lang]
      else
        @lang = BOAST::get_lang
      end
    end

    def print
      @code.rewind
      puts @code.read
    end

    def to_str
      @code.rewind
      return code.read
    end

    def to_s
      @code.rewind
      return code.read
    end

    def setup_compiler(options = {})
      Rake::Task::clear
      verbose = options[:verbose]
      verbose = BOAST::get_verbose if not verbose
      Rake::verbose(verbose)
      Rake::FileUtilsExt.verbose_flag=verbose
      f_compiler = options[:FC]
      c_compiler = options[:CC]
      cxx_compiler = options[:CXX]
      cuda_compiler = options[:NVCC]
      f_flags = options[:FCFLAGS]
      f_flags += " -fPIC"
      f_flags += " -fno-second-underscore" if f_compiler == 'g95'
      ld_flags = options[:LDFLAGS]
      cuda_flags = options[:NVCCFLAGS]
      cuda_flags += " --compiler-options '-fPIC'"


      includes = "-I#{RbConfig::CONFIG["archdir"]}"
      includes += " -I#{RbConfig::CONFIG["rubyhdrdir"]} -I#{RbConfig::CONFIG["rubyhdrdir"]}/#{RbConfig::CONFIG["arch"]}"
      includes += " -I#{RbConfig::CONFIG["rubyarchhdrdir"]}" if RbConfig::CONFIG["rubyarchhdrdir"]
      ld_flags += " -L#{RbConfig::CONFIG["libdir"]} #{RbConfig::CONFIG["LIBRUBYARG"]} -lrt"
      ld_flags += " -lcudart" if @lang == BOAST::CUDA
      narray_path = nil
      begin
        spec = Gem::Specification::find_by_name('narray')
        narray_path = spec.full_gem_path
      rescue Gem::LoadError => e
      rescue NoMethodError => e
        spec = Gem::available?('narray')
        if spec then
          require 'narray' 
          narray_path = Gem.loaded_specs['narray'].full_gem_path
        end
      end
      includes += " -I#{narray_path}" if narray_path
      cflags = options[:CFLAGS]
      cxxflags = options[:CXXFLAGS]
      cflags += " -fPIC #{includes}"
      cxxflags += " -fPIC #{includes}"
      cflags += " -DHAVE_NARRAY_H" if narray_path
      fcflags = f_flags
      cudaflags = cuda_flags

      if options[:openmp] then
        case @lang
        when BOAST::C
          openmp_c_flags = BOAST::get_openmp_flags[c_compiler]
          if not openmp_c_flags then
            keys = BOAST::get_openmp_flags.keys
            keys.each { |k|
              openmp_c_flags = BOAST::get_openmp_flags[k] if c_compiler.match(k)
            }
          end
          raise "unkwown openmp flags for: #{c_compiler}" if not openmp_c_flags
          cflags += " #{openmp_c_flags}"
          openmp_cxx_flags = BOAST::get_openmp_flags[cxx_compiler]
          if not openmp_cxx_flags then
            keys = BOAST::get_openmp_flags.keys
            keys.each { |k|
              openmp_cxx_flags = BOAST::get_openmp_flags[k] if cxx_compiler.match(k)
            }
          end
          raise "unkwown openmp flags for: #{cxx_compiler}" if not openmp_cxx_flags
          cxxflags += " #{openmp_cxx_flags}"
        when BOAST::FORTRAN
          openmp_f_flags = BOAST::get_openmp_flags[f_compiler]
          if not openmp_f_flags then
            keys = BOAST::get_openmp_flags.keys
            keys.each { |k|
              openmp_f_flags = BOAST::get_openmp_flags[k] if f_compiler.match(k)
            }
          end
          raise "unkwown openmp flags for: #{f_compiler}" if not openmp_f_flags
          fcflags += " #{openmp_f_flags}"
        end
      end

      runner = lambda { |t, call_string|
        if verbose then
          sh call_string
        else
          status, stdout, stderr = systemu call_string
          if not status.success? then
            puts stderr
            fail "#{t.source}: compilation failed"
          end
          status.success?
        end
      }

      rule '.o' => '.c' do |t|
        c_call_string = "#{c_compiler} #{cflags} -c -o #{t.name} #{t.source}"
        runner.call(t, c_call_string)
      end

      rule '.o' => '.f90' do |t|
        f_call_string = "#{f_compiler} #{fcflags} -c -o #{t.name} #{t.source}"
        runner.call(t, f_call_string)
      end

      rule '.o' => '.cpp' do |t|
        cxx_call_string = "#{cxx_compiler} #{cxxflags} -c -o #{t.name} #{t.source}"
        runner.call(t, cxx_call_string)
      end

      rule '.o' => '.cu' do |t|
        cuda_call_string = "#{cuda_compiler} #{cudaflags} -c -o #{t.name} #{t.source}"
        runner.call(t, cuda_call_string)
      end
      return ld_flags
    end

    def build_opencl(options)
      require 'opencl_ruby_ffi'
      platform = nil
      platforms = OpenCL::get_platforms
      if options[:platform_vendor] then
        platforms.each{ |p|
          platform = p if p.vendor.match(options[:platform_vendor])
        }
      else
        platform = platforms.first
      end
      device = nil
      type = options[:device_type] ? options[:device_type] : OpenCL::Device::Type::ALL
      devices = platform.devices(type)
      if options[:device_name] then
        devices.each{ |d|
          device = d if d.name.match(options[:device_name])
        }
      else
        device = devices.first
      end
      @context = OpenCL::create_context([device])
      program = @context.create_program_with_source([@code.string])
      opts = options[:CLFLAGS]
      program.build(:options => options[:CLFLAGS])
      if options[:verbose] then
        program.build_log.each {|dev,log|
          STDERR.puts "#{device.name}: #{log}"
        }
      end
      @queue = @context.create_command_queue(device, :properties => OpenCL::CommandQueue::PROFILING_ENABLE)
      @kernel = program.create_kernel(@procedure.name)
      run_method = <<EOF
def self.run(*args)
  raise "Wrong number of arguments \#{args.length} for #{@procedure.parameters.length}" if args.length > #{@procedure.parameters.length+1} or args.length < #{@procedure.parameters.length}
  params = []
  opts = {}
  opts = args.pop if args.length == #{@procedure.parameters.length+1}
  @procedure.parameters.each_index { |i|
    if @procedure.parameters[i].dimension then
      if @procedure.parameters[i].direction == :in and @procedure.parameters[i].direction == :out then
        params[i] = @context.create_buffer( args[i].size * args[i].element_size )
        @queue.enqueue_write_buffer( params[i], args[i], :blocking => true )
      elsif @procedure.parameters[i].direction == :in then
        params[i] = @context.create_buffer( args[i].size * args[i].element_size, :flags => OpenCL::Mem::Flags::READ_ONLY )
        @queue.enqueue_write_buffer( params[i], args[i], :blocking => true )
      elsif @procedure.parameters[i].direction == :out then
        params[i] = @context.create_buffer( args[i].size * args[i].element_size, :flags => OpenCL::Mem::Flags::WRITE_ONLY )
      else
        params[i] = @context.create_buffer( args[i].size * args[i].element_size )
      end
    else
      if @procedure.parameters[i].type.is_a?(Real) then
        params[i] = OpenCL::Half::new(args[i]) if @procedure.parameters[i].type.size == 2
        params[i] = OpenCL::Float::new(args[i]) if @procedure.parameters[i].type.size == 4
        params[i] = OpenCL::Double::new(args[i]) if @procedure.parameters[i].type.size == 8
      elsif @procedure.parameters[i].type.is_a?(Int) then
        if @procedure.parameters[i].type.signed
          params[i] = OpenCL::Char::new(args[i]) if @procedure.parameters[i].type.size == 1
          params[i] = OpenCL::Short::new(args[i]) if @procedure.parameters[i].type.size == 2
          params[i] = OpenCL::Int::new(args[i]) if @procedure.parameters[i].type.size == 4
          params[i] = OpenCL::Long::new(args[i]) if @procedure.parameters[i].type.size == 8
        else
          params[i] = OpenCL::UChar::new(args[i]) if @procedure.parameters[i].type.size == 1
          params[i] = OpenCL::UShort::new(args[i]) if @procedure.parameters[i].type.size == 2
          params[i] = OpenCL::UInt::new(args[i]) if @procedure.parameters[i].type.size == 4
          params[i] = OpenCL::ULong::new(args[i]) if @procedure.parameters[i].type.size == 8
        end
      else
        params[i] = args[i]
      end
    end
  }
  params.each_index{ |i|
    @kernel.set_arg(i, params[i])
  }
  event = @queue.enqueue_NDrange_kernel(@kernel, opts[:global_work_size], :local_work_size => opts[:local_work_size])
  @procedure.parameters.each_index { |i|
    if @procedure.parameters[i].dimension then
      if @procedure.parameters[i].direction == :in and @procedure.parameters[i].direction == :out then
        @queue.enqueue_read_buffer( params[i], args[i], :blocking => true )
      elsif @procedure.parameters[i].direction == :out then
        @queue.enqueue_read_buffer( params[i], args[i], :blocking => true )
      end
    end
  }
  result = {}
  result[:start] = event.profiling_command_start
  result[:end] = event.profiling_command_end
  result[:duration] = (result[:end] - result[:start])/1000000000.0
  return result
end
EOF
    eval run_method
    return self
    end

    def build(options = {})
      compiler_options = BOAST::get_compiler_options
      compiler_options.update(options)
      return build_opencl(compiler_options) if @lang == BOAST::CL
      ldflags = self.setup_compiler(compiler_options)
      extension = ".c" if @lang == BOAST::C
      extension = ".cu" if @lang == BOAST::CUDA
      extension = ".f90" if @lang == BOAST::FORTRAN
#temporary
      c_compiler = compiler_options[:CC]
      c_compiler = "cc" if not c_compiler
      linker = compiler_options[:LD]
      linker = c_compiler if not linker
#end temporary
      if options[:openmp] then
        openmp_ld_flags = BOAST::get_openmp_flags[linker]
          if not openmp_ld_flags then
            keys = BOAST::get_openmp_flags.keys
            keys.each { |k|
              openmp_ld_flags = BOAST::get_openmp_flags[k] if linker.match(k)
            }
          end
          raise "unkwown openmp flags for: #{linker}" if not openmp_ld_flags
          ldflags += " #{openmp_ld_flags}"
      end
      source_file = Tempfile::new([@procedure.name,extension])
      path = source_file.path
      target = path.chomp(File::extname(path))+".o"
      fill_code(source_file)
      source_file.close

      previous_lang = BOAST::get_lang
      previous_output = BOAST::get_output
      BOAST::set_lang(BOAST::C)
      module_file_name = File::split(path.chomp(File::extname(path)))[0] + "/Mod_" + File::split(path.chomp(File::extname(path)))[1].gsub("-","_") + ".c"
      module_name = File::split(module_file_name.chomp(File::extname(module_file_name)))[1]
      module_file = File::open(module_file_name,"w+")
      BOAST::set_output(module_file)
      fill_module(module_file, module_name)
      module_file.rewind
#     puts module_file.read
      module_file.close
      BOAST::set_lang(previous_lang)
      BOAST::set_output(previous_output)
      module_target = module_file_name.chomp(File::extname(module_file_name))+".o"
      module_final = module_file_name.chomp(File::extname(module_file_name))+".so"
      kernel_files = []
      @kernels.each { |kernel|
        kernel_file = Tempfile::new([kernel.procedure.name,".o"])
        kernel.binary.rewind
        kernel_file.write( kernel.binary.read )
        kernel_file.close
        kernel_files.push(kernel_file)
      }
      file module_final => [module_target, target] do
        #puts "#{linker} -shared -o #{module_final} #{module_target} #{target} #{kernel_files.join(" ")} -Wl,-Bsymbolic-functions -Wl,-z,relro -rdynamic -Wl,-export-dynamic #{ldflags}"
        sh "#{linker} -shared -o #{module_final} #{module_target} #{target} #{(kernel_files.collect {|f| f.path}).join(" ")} -Wl,-Bsymbolic-functions -Wl,-z,relro -rdynamic -Wl,-export-dynamic #{ldflags}"
      end
      Rake::Task[module_final].invoke
      require(module_final)
      eval "self.extend(#{module_name})"
      f = File::open(target,"rb")
      @binary = StringIO::new
      @binary.write( f.read )
      f.close
      File.unlink(target)
      File.unlink(module_target)
      File.unlink(module_file_name)
      File.unlink(module_final)
      kernel_files.each { |f|
        f.unlink
      }
      return self
    end

    def fill_code(source_file)
      @code.rewind
      source_file.puts "#include <inttypes.h>" if @lang == BOAST::C or @lang == BOAST::CUDA
      source_file.puts "#include <cuda.h>" if @lang == BOAST::CUDA
      source_file.write @code.read
      if @lang == BOAST::CUDA then
        source_file.write <<EOF
extern "C" {
  #{@procedure.header(BOAST::CUDA,false)}{
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
      @code.rewind
    end

    def fill_module(module_file, module_name)
      module_file.write <<EOF
#include "ruby.h"
#include <inttypes.h>
#include <time.h>
#ifdef HAVE_NARRAY_H
#include "narray.h"
#endif
EOF
      if( @lang == BOAST::CUDA ) then
        module_file.print "#include <cuda_runtime.h>\n"
      end
      module_file.print @procedure.header(@lang)
      module_file.write <<EOF
VALUE #{module_name} = Qnil;
void Init_#{module_name}();
VALUE method_run(int argc, VALUE *argv, VALUE self);
void Init_#{module_name}() {
  #{module_name} = rb_define_module("#{module_name}");
  rb_define_method(#{module_name}, "run", method_run, -1);
}
VALUE method_run(int argc, VALUE *argv, VALUE self) {
EOF
      if( @lang == BOAST::CUDA ) then
        module_file.write <<EOF
  if( argc < #{@procedure.parameters.length} || argc > #{@procedure.parameters.length + 1} )
    rb_raise(rb_eArgError, "wrong number of arguments for #{@procedure.name} (%d for #{@procedure.parameters.length})", argc);
  VALUE rb_opts;
  VALUE rb_ptr;
  size_t block_size[3] = {1,1,1};
  size_t block_number[3] = {1,1,1};
EOF
      else
        module_file.write <<EOF
  if( argc != #{@procedure.parameters.length} )
    rb_raise(rb_eArgError, "wrong number of arguments for #{@procedure.name} (%d for #{@procedure.parameters.length})", argc);
  VALUE rb_ptr;
EOF
      end
      argc = @procedure.parameters.length
      argv = Variable::new("argv",Real,{:dimension => [ Dimension::new(0,argc-1) ] })
      rb_ptr = Variable::new("rb_ptr",Int)
      @procedure.parameters.each { |param| 
        param_copy = param.copy
        param_copy.constant = nil
        param_copy.direction = nil
        param_copy.decl
      }
      @procedure.parameters.each_index do |i|
        param = @procedure.parameters[i]
        if not param.dimension then
          case param.type
            when Int 
              (param === FuncCall::new("NUM2INT", argv[i])).print if param.type.size == 4
              (param === FuncCall::new("NUM2LONG", argv[i])).print if param.type.size == 8
            when Real
              (param === FuncCall::new("NUM2DBL", argv[i])).print
          end
        else
          (rb_ptr === argv[i]).print
          if @lang == BOAST::CUDA then
            module_file.print <<EOF
  if ( IsNArray(rb_ptr) ) {
    struct NARRAY *n_ary;
    size_t array_size;
    Data_Get_Struct(rb_ptr, struct NARRAY, n_ary);
    array_size = n_ary->total * na_sizeof[n_ary->type];
    cudaMalloc( (void **) &#{param.name}, array_size);
EOF
            if param.direction == :in then
            module_file.print <<EOF
    cudaMemcpy(#{param.name}, (void *) n_ary->ptr, array_size, cudaMemcpyHostToDevice);
EOF
            end
            module_file.print <<EOF
  } else
    rb_raise(rb_eArgError, "wrong type of argument %d", #{i});
  
EOF
          else
            module_file.print <<EOF
  if (TYPE(rb_ptr) == T_STRING) {
    #{param.name} = (void *) RSTRING_PTR(rb_ptr);
  } else if ( IsNArray(rb_ptr) ) {
    struct NARRAY *n_ary;
    Data_Get_Struct(rb_ptr, struct NARRAY, n_ary);
    #{param.name} = (void *) n_ary->ptr;
  } else
    rb_raise(rb_eArgError, "wrong type of argument %d", #{i});
EOF
          end
        end
      end
      if @lang == BOAST::CUDA then
        module_file.write <<EOF
  if( argc == #{@procedure.parameters.length + 1} ) {
    rb_opts = argv[argc -1];
    if ( rb_opts != Qnil ) {
      VALUE rb_array_data = Qnil;
      int i;
      if (TYPE(rb_opts) != T_HASH)
        rb_raise(rb_eArgError, "Cuda options should be passed as a hash");
      rb_ptr = rb_hash_aref(rb_opts, ID2SYM(rb_intern("block_size")));
      if( rb_ptr != Qnil ) {
        if (TYPE(rb_ptr) != T_ARRAY)
          rb_raise(rb_eArgError, "Cuda option block_size should be an array");
        for(i=0; i<3; i++) {
          rb_array_data = rb_ary_entry(rb_ptr, i);
          if( rb_array_data != Qnil )
            block_size[i] = (size_t) NUM2LONG( rb_array_data );
        }
      }
      rb_ptr = rb_hash_aref(rb_opts, ID2SYM(rb_intern("block_number")));
      if( rb_ptr != Qnil ) {
        if (TYPE(rb_ptr) != T_ARRAY)
          rb_raise(rb_eArgError, "Cuda option block_number should be an array");
        for(i=0; i<3; i++) {
          rb_array_data = rb_ary_entry(rb_ptr, i);
          if( rb_array_data != Qnil )
            block_number[i] = (size_t) NUM2LONG( rb_array_data );
        }
      }
    }
  }
EOF
      end
      module_file.print "  #{@procedure.properties[:return].type.decl} ret;\n" if @procedure.properties[:return]
      module_file.print "  VALUE stats = rb_hash_new();\n"
      module_file.print "  struct timespec start, stop;\n"
      module_file.print "  unsigned long long int duration;\n"
      module_file.print "  clock_gettime(CLOCK_REALTIME, &start);\n"
      if @lang == BOAST::CUDA then
        module_file.print "  duration = "
      elsif @procedure.properties[:return] then
        module_file.print "  ret = "
      end
      module_file.print "  #{@procedure.name}"
      module_file.print "_" if @lang == BOAST::FORTRAN
      module_file.print "_wrapper" if @lang == BOAST::CUDA
      module_file.print "("
      if(@lang == BOAST::FORTRAN) then
        params = []
        @procedure.parameters.each { |param|
          if param.dimension then
            params.push( param.name )
          else
            params.push( "&"+param.name )
          end
        }
        module_file.print params.join(", ")
      else
        module_file.print @procedure.parameters.join(", ") 
      end
      if @lang == BOAST::CUDA then
        module_file.print ", " if @procedure.parameters.length > 0
        module_file.print "block_number, block_size"
      end
      module_file.print "  );\n"
      module_file.print "  clock_gettime(CLOCK_REALTIME, &stop);\n"

      if @lang == BOAST::CUDA then
        @procedure.parameters.each_index do |i|
          param = @procedure.parameters[i]
          if param.dimension then
            (rb_ptr === argv[i]).print
            module_file.print <<EOF
  if ( IsNArray(rb_ptr) ) {
EOF
            if param.direction == :out then
            module_file.print <<EOF
    struct NARRAY *n_ary;
    size_t array_size;
    Data_Get_Struct(rb_ptr, struct NARRAY, n_ary);
    array_size = n_ary->total * na_sizeof[n_ary->type];
    cudaMemcpy(#{param.name}, (void *) n_ary->ptr, array_size, cudaMemcpyDeviceToHost);
EOF
            end
            module_file.print <<EOF
    cudaFree( (void *) #{param.name});
  } else
    rb_raise(rb_eArgError, "wrong type of argument %d", #{i});
  
EOF
          end
        end
      end
      if @lang != BOAST::CUDA then
        module_file.print "  duration = (unsigned long long int)stop.tv_sec * (unsigned long long int)1000000000 + stop.tv_nsec;\n"
        module_file.print "  duration -= (unsigned long long int)start.tv_sec * (unsigned long long int)1000000000 + start.tv_nsec;\n"
      end
      module_file.print "  rb_hash_aset(stats,ID2SYM(rb_intern(\"duration\")),rb_float_new((double)duration*(double)1e-9));\n"
      if @procedure.properties[:return] then
        type_ret = @procedure.properties[:return].type
        module_file.print "  rb_hash_aset(stats,ID2SYM(rb_intern(\"return\")),rb_int_new((long long)ret));\n" if type_ret.kind_of?(Int) and type_ret.signed
        module_file.print "  rb_hash_aset(stats,ID2SYM(rb_intern(\"return\")),rb_int_new((unsigned long long)ret));\n" if type_ret.kind_of?(Int) and not type_ret.signed
        module_file.print "  rb_hash_aset(stats,ID2SYM(rb_intern(\"return\")),rb_float_new((double)ret));\n" if type_ret.kind_of?(Real)
      end
      module_file.print "  return stats;\n"
      module_file.print  "}"
    end

    def method_missing(meth, *args, &block)
     if meth.to_s == "run" then
       self.build
       self.run(*args,&block)
     else
       super
     end
    end
  end
end
