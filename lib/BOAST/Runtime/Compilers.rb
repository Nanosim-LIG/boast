require 'rake'
require 'rbconfig'
require 'systemu'
require 'os'

module BOAST

  # @private
  module Compilers
    include Rake::DSL

    def get_openmp_flags(compiler)
      openmp_flags = BOAST::get_openmp_flags[compiler]
      unless openmp_flags then
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
        narray_path = spec.require_path
        unless File.exist?(narray_path+"/narray.h") then
          narray_path = spec.full_gem_path
        end
      rescue Gem::LoadError => e
      rescue NoMethodError => e
        spec = Gem::available?('narray')
        if spec then
          require 'narray'
          narray_path = Gem.loaded_specs['narray'].require_path
          unless File.exist?(narray_path+"/narray.h") then
            narray_path = Gem.loaded_specs['narray'].full_gem_path
          end
        end
      end
      return narray_path
    end

    def setup_c_compiler(options, includes, narray_path, runner, probes)
      c_mppa_compiler = "k1-gcc"
      c_compiler = options[:CC]
      cflags = options[:CFLAGS]
      probes.each { |p|
        cflags += " #{p.cflags}" if p.respond_to?(:cflags)
      }
      if @architecture == SPARC || @architecture = PPC then
        cflags += " -mcpu=#{get_model}"
      else
        cflags += " -march=#{get_model}"
      end if
      cflags += " -fPIC #{includes}"
      cflags += " -DHAVE_NARRAY_H" if narray_path
      cflags += " -I/usr/local/k1tools/include" if @architecture == MPPA
      objext = RbConfig::CONFIG["OBJEXT"]
      if (options[:openmp] or options[:OPENMP]) and @lang == C and not disable_openmp then
          openmp_cflags = get_openmp_flags(c_compiler)
          raise "unkwown openmp flags for: #{c_compiler}" unless openmp_cflags
          cflags += " #{openmp_cflags}"
      end

      cflags_no_fpic = cflags.gsub("-fPIC","")

      rule ".nofpic#{objext}" => '.c' do |t|
        c_call_string = "#{c_compiler} #{cflags_no_fpic} -c -o #{t.name} #{t.source}"
        runner.call(t, c_call_string)
      end

      rule ".#{objext}" => '.c' do |t|
        c_call_string = "#{c_compiler} #{cflags} -c -o #{t.name} #{t.source}"
        runner.call(t, c_call_string)
      end

      rule ".#{objext}io" => ".cio" do |t|
        c_call_string = "#{c_mppa_compiler} -mcore=k1bio -mos=rtems"
        c_call_string += " -mboard=developer -x c -c -o #{t.name} #{t.source}"
        runner.call(t, c_call_string)
      end

      rule ".#{objext}comp" => ".ccomp" do |t|
        c_call_string = "#{c_mppa_compiler} -mcore=k1bdp -mos=nodeos"
        c_call_string += " -mboard=developer -x c -c -o #{t.name} #{t.source}"
        runner.call(t, c_call_string)
      end
    end

    def setup_cxx_compiler(options, includes, runner, probes)
      cxx_compiler = options[:CXX]
      cxxflags = options[:CXXFLAGS]
      cxxflags += " -fPIC #{includes}"
      if (options[:openmp] or options[:OPENMP]) and @lang == C and not disable_openmp then
          openmp_cxxflags = get_openmp_flags(cxx_compiler)
          raise "unkwown openmp flags for: #{cxx_compiler}" unless openmp_cxxflags
          cxxflags += " #{openmp_cxxflags}"
      end

      cxxflags_no_fpic = cxxflags.gsub("-fPIC","")

      rule ".nofpic#{RbConfig::CONFIG["OBJEXT"]}" => '.cpp' do |t|
        cxx_call_string = "#{cxx_compiler} #{cxxflags_no_fpic} -c -o #{t.name} #{t.source}"
        runner.call(t, cxx_call_string)
      end

      rule ".#{RbConfig::CONFIG["OBJEXT"]}" => '.cpp' do |t|
        cxx_call_string = "#{cxx_compiler} #{cxxflags} -c -o #{t.name} #{t.source}"
        runner.call(t, cxx_call_string)
      end
    end

    def setup_fortran_compiler(options, runner, probes)
      f_compiler = options[:FC]
      fcflags = options[:FCFLAGS]
      if @architecture == SPARC || @architecture = PPC then
        fcflags += " -mcpu=#{get_model}"
      else
        fcflags += " -march=#{get_model}"
      end if
      fcflags += " -fPIC"
      fcflags += " -fno-second-underscore" if f_compiler == 'g95'
      if (options[:openmp] or options[:OPENMP]) and @lang == FORTRAN and not disable_openmp then
          openmp_fcflags = get_openmp_flags(f_compiler)
          raise "unkwown openmp flags for: #{f_compiler}" unless openmp_fcflags
          fcflags += " #{openmp_fcflags}"
      end

      fcflags_no_fpic = fcflags.gsub("-fPIC","")

      rule ".nofpic#{RbConfig::CONFIG["OBJEXT"]}" => '.f90' do |t|
        f_call_string = "#{f_compiler} #{fcflags_no_fpic} -c -o #{t.name} #{t.source}"
        runner.call(t, f_call_string)
      end

      rule ".#{RbConfig::CONFIG["OBJEXT"]}" => '.f90' do |t|
        f_call_string = "#{f_compiler} #{fcflags} -c -o #{t.name} #{t.source}"
        runner.call(t, f_call_string)
      end
    end

    def setup_cuda_compiler(options, runner, probes)
      cuda_compiler = options[:NVCC]
      cudaflags = options[:NVCCFLAGS]
      cudaflags += " --compiler-options '-fPIC','-D_FORCE_INLINES'"

      rule ".#{RbConfig::CONFIG["OBJEXT"]}" => '.cu' do |t|
        cuda_call_string = "#{cuda_compiler} #{cudaflags} -c -o #{t.name} #{t.source}"
        runner.call(t, cuda_call_string)
      end
    end

    def setup_linker_mppa(options, runner, probes)
      objext = RbConfig::CONFIG["OBJEXT"]
      ldflags = options[:LDFLAGS]
      board = " -mboard=developer"
      ldflags += " -lmppaipc"
      
      linker = "k1-gcc"
      
      rule ".bincomp" => ".#{objext}comp" do |t|
        linker_string = "#{linker} -o #{t.name} #{t.source} -mcore=k1bdp #{board} -mos=nodeos #{ldflags}"
        runner.call(t, linker_string)
      end
      
      rule ".binio" => ".#{objext}io" do |t|
        linker_string = "#{linker} -o #{t.name} #{t.source} -mcore=k1bio #{board} -mos=rtems #{ldflags}"
        runner.call(t, linker_string)
      end

    end

    def setup_linker(options, probes)
      ldflags = options[:LDFLAGS]
      if @architecture == SPARC || @architecture = PPC then
        ldflags += " -mcpu=#{get_model}"
      else
        ldflags += " -march=#{get_model}"
      end if
      ldflags += " -L#{RbConfig::CONFIG["libdir"]}"
      if RbConfig::CONFIG["ENABLE_SHARED"] != "no" then
        ldflags += " #{RbConfig::CONFIG["LIBRUBYARG"]}"
      else
        ldflags += " -Wl,-R#{RbConfig::CONFIG["libdir"]}"
      end
      probes.each { |p|
        ldflags += " #{p.ldflags}" if p.respond_to?(:ldflags)
      }
      ldflags += " -lcudart" if @lang == CUDA
      ldflags += " -L/usr/local/k1tools/lib64 -lmppaipc -lpcie -lz -lelf -lmppa_multiloader" if @architecture == MPPA
      ldflags += " -lmppamon -lmppabm -lm -lmppalock" if @architecture == MPPA
      c_compiler = options[:CC]
      c_compiler = "cc" unless c_compiler
      linker = options[:LD]
      linker = c_compiler unless linker
      if (options[:openmp] or options[:OPENMP]) and not disable_openmp then
        openmp_ldflags = get_openmp_flags(linker)
        raise "unknown openmp flags for: #{linker}" unless openmp_ldflags
        ldflags += " #{openmp_ldflags}"
      end

      if OS.mac? then
        ldshared = "-dynamic -bundle"
        ldshared_flags = "-Wl,-undefined,dynamic_lookup -Wl,-multiply_defined,suppress"
      elsif @architecture == SPARC then
        ldshared = "-shared"
        ldshared_flags = "-Wl,-Bsymbolic -Wl,-z,relro -Wl,-export-dynamic"
      else
        ldshared = "-shared"
        ldshared_flags = "-Wl,-Bsymbolic-functions -Wl,-z,relro -rdynamic -Wl,-export-dynamic"
      end

      return [linker, ldshared, ldshared_flags, ldflags]
    end

    def setup_compilers(probes, options = {})
      Rake::Task::clear
      Rake::verbose(false)
      Rake::FileUtilsExt.verbose_flag=false

      narray_path = get_narray_path
      includes = get_includes(narray_path)

      runner = lambda { |t, call_string|
        puts call_string if get_verbose
        status, stdout, stderr = systemu call_string
        if get_verbose
          puts stdout.force_encoding("UTF-8") if stdout != ""
          puts stderr.force_encoding("UTF-8") if stderr != ""
        end
        unless status.success? then
          puts stderr.force_encoding("UTF-8") unless get_verbose
          fail "#{t.source}: compilation failed"
        end
        status.success?
      }

      setup_c_compiler(options, includes, narray_path, runner, probes)
      setup_cxx_compiler(options, includes, runner, probes)
      setup_fortran_compiler(options, runner, probes)
      setup_cuda_compiler(options, runner, probes)
      
      setup_linker_mppa(options, runner, probes) if @architecture == MPPA

      return setup_linker(options, probes)

    end

  end

end
