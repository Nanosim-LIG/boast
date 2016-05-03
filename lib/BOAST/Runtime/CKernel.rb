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
  end
end
