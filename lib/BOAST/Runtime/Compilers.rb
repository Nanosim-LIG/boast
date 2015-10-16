require 'rake'
require 'rbconfig'
require 'systemu'
require 'os'

module BOAST

  module Compilers
    include Rake::DSL

    def get_openmp_flags(compiler)
      openmp_flags = BOAST::get_openmp_flags[compiler]
      if not openmp_flags then
        keys = BOAST::get_openmp_flags.keys
        keys.each { |k|
          openmp_flags = BOAST::get_openmp_flags[k] if compiler.match(k)
        }
      end
      return openmp_flags
    end

    def get_includes(narray_path)
      includes = "-I#{RbConfig::CONFIG["archdir"]}"
      includes += " -I#{RbConfig::CONFIG["rubyhdrdir"]} -I#{RbConfig::CONFIG["rubyhdrdir"]}/#{RbConfig::CONFIG["arch"]}"
      includes += " -I#{RbConfig::CONFIG["rubyarchhdrdir"]}" if RbConfig::CONFIG["rubyarchhdrdir"]
      includes += " -I#{narray_path}" if narray_path
      return includes
    end

    def get_narray_path
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
    end

    def setup_c_compiler(options, includes, narray_path, runner)
      c_mppa_compiler = "k1-gcc"
      c_compiler = options[:CC]
      cflags = options[:CFLAGS]
      cflags += " -march=#{get_model}"
      cflags += " -fPIC #{includes}"
      cflags += " -DHAVE_NARRAY_H" if narray_path
      cflags += " -I/usr/local/k1tools/include" if @architecture == MPPA
      objext = RbConfig::CONFIG["OBJEXT"]
      if options[:openmp] and @lang == C then
          openmp_cflags = get_openmp_flags(c_compiler)
          raise "unkwown openmp flags for: #{c_compiler}" if not openmp_cflags
          cflags += " #{openmp_cflags}"
      end

      rule ".#{objext}" => '.c' do |t|
        c_call_string = "#{c_compiler} #{cflags} -c -o #{t.name} #{t.source}"
        runner.call(t, c_call_string)
      end

      rule ".#{objext}io" => ".cio" do |t|
        c_call_string = "#{c_mppa_compiler} -mcore=k1io -mos=rtems"
        c_call_string += " -mboard=developer -x c -c -o #{t.name} #{t.source}"
        runner.call(t, c_call_string)
      end

      rule ".#{objext}comp" => ".ccomp" do |t|
        c_call_string = "#{c_mppa_compiler} -mcore=k1dp -mos=nodeos"
        c_call_string += " -mboard=developer -x c -c -o #{t.name} #{t.source}"
        runner.call(t, c_call_string)
      end
    end

    def setup_cxx_compiler(options, includes, runner)
      cxx_compiler = options[:CXX]
      cxxflags = options[:CXXFLAGS]
      cxxflags += " -fPIC #{includes}"
      if options[:openmp] and @lang == C then
          openmp_cxxflags = get_openmp_flags(cxx_compiler)
          raise "unkwown openmp flags for: #{cxx_compiler}" if not openmp_cxxflags
          cxxflags += " #{openmp_cxxflags}"
      end

      rule ".#{RbConfig::CONFIG["OBJEXT"]}" => '.cpp' do |t|
        cxx_call_string = "#{cxx_compiler} #{cxxflags} -c -o #{t.name} #{t.source}"
        runner.call(t, cxx_call_string)
      end
    end

    def setup_fortran_compiler(options, runner)
      f_compiler = options[:FC]
      fcflags = options[:FCFLAGS]
      fcflags += " -march=#{get_model}"
      fcflags += " -fPIC"
      fcflags += " -fno-second-underscore" if f_compiler == 'g95'
      if options[:openmp] and @lang == FORTRAN then
          openmp_fcflags = get_openmp_flags(f_compiler)
          raise "unkwown openmp flags for: #{f_compiler}" if not openmp_fcflags
          fcflags += " #{openmp_fcflags}"
      end

      rule ".#{RbConfig::CONFIG["OBJEXT"]}" => '.f90' do |t|
        f_call_string = "#{f_compiler} #{fcflags} -c -o #{t.name} #{t.source}"
        runner.call(t, f_call_string)
      end
    end

    def setup_cuda_compiler(options, runner)
      cuda_compiler = options[:NVCC]
      cudaflags = options[:NVCCFLAGS]
      cudaflags += " --compiler-options '-fPIC'"

      rule ".#{RbConfig::CONFIG["OBJEXT"]}" => '.cu' do |t|
        cuda_call_string = "#{cuda_compiler} #{cudaflags} -c -o #{t.name} #{t.source}"
        runner.call(t, cuda_call_string)
      end
    end

    def setup_linker_mppa(options, runner)
      objext = RbConfig::CONFIG["OBJEXT"]
      ldflags = options[:LDFLAGS]
      board = " -mboard=developer"
      ldflags += " -lmppaipc"
      
      linker = "k1-gcc"
      
      rule ".bincomp" => ".#{objext}comp" do |t|
        linker_string = "#{linker} -o #{t.name} #{t.source} -mcore=k1dp #{board} -mos=nodeos #{ldflags}"
        runner.call(t, linker_string)
      end
      
      rule ".binio" => ".#{objext}io" do |t|
        linker_string = "#{linker} -o #{t.name} #{t.source} -mcore=k1io #{board} -mos=rtems #{ldflags}"
        runner.call(t, linker_string)
      end

    end

    def setup_linker(options)
      ldflags = options[:LDFLAGS]
      ldflags += " -march=#{get_model}"
      ldflags += " -L#{RbConfig::CONFIG["libdir"]} #{RbConfig::CONFIG["LIBRUBYARG"]}"
      ldflags += " -lrt" if not OS.mac?
      ldflags += " -lcudart" if @lang == CUDA
      ldflags += " -L/usr/local/k1tools/lib64 -lmppaipc -lpcie -lz -lelf -lmppa_multiloader" if @architecture == MPPA
      ldflags += " -lmppamon -lmppabm -lm -lmppalock" if @architecture == MPPA
      c_compiler = options[:CC]
      c_compiler = "cc" if not c_compiler
      linker = options[:LD]
      linker = c_compiler if not linker
      if options[:openmp] then
        openmp_ldflags = get_openmp_flags(linker)
        raise "unknown openmp flags for: #{linker}" if not openmp_ldflags
        ldflags += " #{openmp_ldflags}"
      end

      if OS.mac? then
        ldflags = "-Wl,-undefined,dynamic_lookup -Wl,-multiply_defined,suppress #{ldflags}"
        ldshared = "-dynamic -bundle"
      else
        ldflags = "-Wl,-Bsymbolic-functions -Wl,-z,relro -rdynamic -Wl,-export-dynamic #{ldflags}"
        ldshared = "-shared"
      end

      return [linker, ldshared, ldflags]
    end

    def setup_compilers(options = {})
      Rake::Task::clear
      verbose = options[:verbose]
      verbose = get_verbose if not verbose
      Rake::verbose(verbose)
      Rake::FileUtilsExt.verbose_flag=verbose

      narray_path = get_narray_path
      includes = get_includes(narray_path)

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

      setup_c_compiler(options, includes, narray_path, runner)
      setup_cxx_compiler(options, includes, runner)
      setup_fortran_compiler(options, runner)
      setup_cuda_compiler(options, runner)
      
      setup_linker_mppa(options, runner) if @architecture == MPPA

      return setup_linker(options)

    end

  end

end
