require 'stringio'
require 'rake'
require 'tempfile'
require 'rbconfig'
require 'systemu'
require 'yaml'
require 'pathname'
require 'os'
require 'narray_ffi'

module BOAST

  class CKernel
    include Compilers
    include Rake::DSL
    include Inspectable
    include PrivateStateAccessor
    include TypeTransition

    attr_accessor :code
    attr_accessor :procedure
    attr_accessor :lang
    attr_accessor :architecture
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
      @probes.push AffinityProbe unless OS.mac?

      case @lang
      when CL
        extend OpenCLRuntime
      when CUDA
        extend CUDARuntime
        @probes = []
      when FORTRAN
        extend FORTRANRuntime
        extend FFIRuntime if ffi?
      else
        if @architecture == MPPA then
          extend MPPARuntime
          @probes = [MPPAProbe]
        else
          extend CRuntime
          extend FFIRuntime if ffi?
        end
      end
    end

    def print
      @code.rewind
      puts @code.read
    end

    def to_s
      if @lang == FORTRAN then
        return line_limited_source
      else
        @code.rewind
        return code.read
      end
    end

    # @private
    def method_missing(meth, *args, &block)
     if meth.to_s == "run" then
       build
       run(*args, &block)
     else
       super
     end
    end

    def cost(*args)
      @cost_function.call(*args)
    end

    # @!method build( options = {} )
    # Builds the computing kernel.
    # @param [Hash] options contain build time options. Usual compiling flags are supported. Default values can be overriden in $XDG_CONFIG_HOME/.config/BOAST/compiler_options or $HOME/.config/BOAST/compiler_options. The same flags can be set as environment variables. Flags given here override environment variable ones.
    # @option options [String] :CC C compiler
    # @option options [String] :CFLAGS C compiler flags
    # @option options [String] :FC Fortran compiler
    # @option options [String] :FCFLAGS Fortran compiler flags
    # @option options [String] :CXX C++ compiler
    # @option options [String] :CXXFLAGS C++ compiler flags
    # @option options [String] :LD linker
    # @option options [String] :LDFLAGS linker flags
    # @option options [Boolean] :OPENMP activate OpenMP support. Correct flag should be set for your compiler in $XDG_CONFIG_HOME/.config/BOAST/openmp_flags or $HOME/.config/BOAST/openmp_flags.
    # @option options [String] :NVCC cuda compiler
    # @option options [String] :NVCCFLAGS cuda compiler flags
    # @option options [String] :CLFLAGS opencl compiation flags
    # @option options [String] :CLVENDOR restrict selected OpenCL platforms to the ones which vendor match the option
    # @option options [String] :CLPLATFORM restrict selected OpenCL platforms to the ones which name match the option
    # @option options [String] :CLDEVICE restrict selected OpenCL devices to the ones which mame match the option or use the provided OpenCL::Device
    # @option options [String] :CLCONTEXT use the devices in the given OpenCL::Context
    # @option options [String] :CLDEVICETYPE restrict selected OpenCL devices to the corresponding types
  end
end
