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

    # Creates a new CKernel object. BOAST output is redirected to the CKernel. If the chain_code state is set the current BOAST output, as returned by {BOAST.get_output}, is used.
    # @param [Hash] options contains named options
    # @option options [StringIO] :code specify a StringIO to use rather than create a new one.
    # @option options [Array] :kernels list of kernels this kernel depends on. The kernels will be linked at build time.
    # @option options [Integer] :lang specify the language to use. Default is current language state as returned by {BOAST.get_lang}.
    # @option options [Integer] :architecture specify the architecture to use. Default is the current BOAST architecture as returned by {BOAST.get_architecture}.
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
        else
          extend CRuntime
          extend FFIRuntime if ffi?
        end
      end
    end

    # @deprecated
    def print
      @code.rewind
      puts @code.read
    end

    # @return [String] source code of the kernel
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

    # If a cost function is provided returns the cost of running the function on the provided arguments.
    def cost(*args)
      @cost_function.call(*args)
    end

    # @!method build( options = {} )
    # Builds the computing kernel.
    # @param [Hash] options contains build time options. Usual compiling flags are supported. Default values can be overriden in $XDG_CONFIG_HOME/.config/BOAST/compiler_options or $HOME/.config/BOAST/compiler_options. The same flags can be set as environment variables. Flags given here override environment variable ones.
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

    # @!method run( *args, options = {} )
    # Runs the computing kernel using the given arguments.
    # @param args the arguments corresponding to the list of parameters of the #procedure attribute
    # @param [Hash] options contains runtime options.
    # @option options [Array] :global_work_size only considered for CUDA and OpenCL kernels. See corresponding OpenCL documentation
    # @option options [Array] :local_work_size only considered for CUDA and OpenCL kernels. See corresponding OpenCL documentation
    # @option options [Array] :block_number only considered for CUDA and OpenCL kernels. See corresponding CUDA documentation
    # @option options [Array] :block_size only considered for CUDA and OpenCL kernels. See corresponding CUDA documentation
    # @option options [Array] :PAPI list of PAPI counters to monitor. ( ex: ['PAPI_L1_DCM', 'PAPI_L2_DCM'], see PAPI documentation.
    # @return [Hash] contains at least the *:duration* entry which is the runtime of the kernel in seconds. If the kernel is a function then the *:return* field will contain the returned value. For :inout or :out scalars the *:reference_return* field will be a Hash with each parameter name associated to the corresponding value. If *:PAPI* options was given will contain a *:PAPI* entry with the corresponding counters value.
  end
end
