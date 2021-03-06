module BOAST

  # @private
  module OpenCLRuntime

    attr_reader :context
    attr_reader :queue
    attr_accessor :il

    def il?
      @il
    end

    def select_cl_platforms(options)
      platforms = OpenCL::get_platforms
      if options[:CLVENDOR] then
        platforms.select!{ |p|
          p.vendor.match(options[:CLVENDOR])
        }
      end
      if options[:CLPLATFORM] then
        platforms.select!{ |p|
          p.name.match(options[:CLPLATFORM])
        }
      end
      return platforms
    end

    def select_cl_device(options, context = nil)
      return options[:CLDEVICE] if options[:CLDEVICE] and options[:CLDEVICE].is_a?(OpenCL::Device)
      devices = nil
      if context then
        devices = context.devices
      else
        platforms = select_cl_platforms(options)
        type = options[:CLDEVICETYPE] ? OpenCL::Device::Type.const_get(options[:CLDEVICETYPE]) : OpenCL::Device::Type::ALL
        devices = platforms.collect { |plt| plt.devices(type) }
        devices.flatten!
      end
      name_pattern = options[:CLDEVICE]
      if name_pattern then
        devices.select!{ |d|
          d.name.match(name_pattern)
        }
      end
      return devices.first
    end

    def init_opencl(options)
      require 'opencl_ruby_ffi'
      require 'BOAST/Runtime/OpenCLTypes.rb'
      device = nil
      if options[:CLCONTEXT] then
        @context = options[:CLCONTEXT]
        device = select_cl_device(options, @context)
      else
        device = select_cl_device(options)
        @context = OpenCL::create_context([device])
      end
      if il?
        program = @context.create_program_with_il(@code.string)
      else
        program = @context.create_program_with_source([@code.string])
      end
      begin
        program.build(:options => options[:CLFLAGS])
      rescue OpenCL::Error => e
        puts e.to_s
        puts program.build_status
        puts program.build_log
        if (options[:VERBOSE] or get_verbose) and !@il then
          puts @code.string
        end
        raise "OpenCL Failed to build #{@procedure.name}"
      end
      if options[:VERBOSE] or get_verbose then
        program.build_log.each { |dev,log|
          puts "#{device.name}: #{log}"
        }
      end
      @queue = @context.create_command_queue(device, :properties => OpenCL::CommandQueue::PROFILING_ENABLE)
      @kernel = program.create_kernel(@procedure.name)
      return self
    end

    def create_opencl_array(arg, parameter)
      return arg if arg.kind_of? OpenCL::Mem
      if parameter.direction == :in then
        flags = OpenCL::Mem::Flags::READ_ONLY
      elsif parameter.direction == :out then
        flags = OpenCL::Mem::Flags::WRITE_ONLY
      else
        flags = OpenCL::Mem::Flags::READ_WRITE
      end
      if parameter.texture then
        param = @context.create_image_2D( OpenCL::ImageFormat::new( OpenCL::ChannelOrder::R, OpenCL::ChannelType::UNORM_INT8 ), arg.size * arg.element_size, 1, :flags => flags )
        @queue.enqueue_write_image( param, arg, :blocking => true )
      else
        if arg.kind_of?(String)
          p = FFI::MemoryPointer.from_string(arg)
          param = @context.create_buffer( p.size, flags: OpenCL::Mem::COPY_HOST_PTR, host_ptr: p )
        else
          param = @context.create_buffer( arg.size * arg.element_size, :flags => flags )
          @queue.enqueue_write_buffer( param, arg, :blocking => true )
        end
      end
      return param
    end

    def create_opencl_scalar(arg, parameter)
      if parameter.type.is_a?(Real) then
        return OPENCL_REAL_TYPES[parameter.type.size]::new(arg)
      elsif parameter.type.is_a?(Int) then
        return OPENCL_INT_TYPES[parameter.type.signed][parameter.type.size]::new(arg)
      else
        return arg
      end
    end

    def create_opencl_param(arg, parameter)
      if parameter.dimension then
        return create_opencl_array(arg, parameter)
      else
        return create_opencl_scalar(arg, parameter)
      end
    end

    def read_opencl_param(param, arg, parameter)
      return arg if arg.kind_of? OpenCL::Mem
      if parameter.texture then
        @queue.enqueue_read_image( param, arg, :blocking => true )
      else
        @queue.enqueue_read_buffer( param, arg, :blocking => true )
      end
    end

    def build(options={})
      compiler_options = BOAST::get_compiler_options
      compiler_options.update(options)
      @probes = compiler_options[:probes] if compiler_options[:probes]
      init_opencl(compiler_options)

      run_method = <<EOF
def self.run(*args)
  raise "Wrong number of arguments \#{args.length} for #{@procedure.parameters.length}" if args.length > #{@procedure.parameters.length+1} or args.length < #{@procedure.parameters.length}
  params = []
  opts = BOAST::get_run_config
  opts = opts.update(args.pop) if args.length == #{@procedure.parameters.length+1}
  @procedure.parameters.each_index { |i|
    params[i] = create_opencl_param( args[i], @procedure.parameters[i] )
  }
  params.each_index{ |i|
    @kernel.set_arg(i, params[i])
  }
  gws = opts[:global_work_size]
  unless gws then
    raise ":global_work_size or :block_number are required to run OpenCL kernels!" unless opts[:block_number]
    gws = []
    opts[:block_number].each_index { |i|
      raise "if using :block_number, :block_size is required  to run OpenCL kernels!" unless opts[:block_size]
      gws.push(opts[:block_number][i]*opts[:block_size][i])
    }
  end
  lws = opts[:local_work_size]
  unless lws then
    lws = opts[:block_size]
  end
  gwo = opts[:global_work_offset]
  event1 = @queue.enqueue_NDrange_kernel(@kernel, gws, local_work_size: lws, global_work_offset: gwo)
  if opts[:repeat] and opts[:repeat] > 1 then
    (opts[:repeat] - 2).times {
      @queue.enqueue_NDrange_kernel(@kernel, gws, local_work_size: lws, global_work_offset: gwo)
    }
    event2 = @queue.enqueue_NDrange_kernel(@kernel, gws, local_work_size: lws, global_work_offset: gwo)
  end
  @procedure.parameters.each_index { |i|
    if @procedure.parameters[i].dimension and (@procedure.parameters[i].direction == :inout or @procedure.parameters[i].direction == :out) then
      read_opencl_param( params[i], args[i], @procedure.parameters[i] )
    end
  }
  @queue.finish
  start_t = event1.profiling_command_start
  if opts[:repeat] and opts[:repeat] > 1 then
    end_t = event2.profiling_command_end
  else
    end_t = event1.profiling_command_end
  end
  result = {}
  result[:start] = start_t
  result[:end] = end_t
  result[:duration] = (end_t - start_t)/1000000000.0
  return result
end
EOF
      eval run_method
      return self
    end

  end

end
