module BOAST
  module OpenCLRuntime

    attr_reader :context
    attr_reader :queue

    def select_cl_platforms(options)
      platforms = OpenCL::get_platforms
      if options[:platform_vendor] then
        platforms.select!{ |p|
          p.vendor.match(options[:platform_vendor])
        }
      elsif options[:CLVENDOR] then
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

    def select_cl_device(options)
      return options[:CLDEVICE] if options[:CLDEVICE] and options[:CLDEVICE].is_a?(OpenCL::Device)

      platforms = select_cl_platforms(options)
      type = options[:device_type] ? OpenCL::Device::Type.const_get(options[:device_type]) : options[:CLDEVICETYPE] ? OpenCL::Device::Type.const_get(options[:CLDEVICETYPE]) : OpenCL::Device::Type::ALL
      devices = platforms.collect { |plt| plt.devices(type) }
      devices.flatten!
      if options[:device_name] then
        devices.select!{ |d|
          d.name.match(options[:device_name])
        }
      elsif options[:CLDEVICE] then
        devices.select!{ |d|
          d.name.match(options[:CLDEVICE])
        }
      end
      return devices.first
    end

    def init_opencl_types
      @@opencl_real_types = {
        2 => OpenCL::Half1,
        4 => OpenCL::Float1,
        8 => OpenCL::Double1
      }

      @@opencl_int_types = {
        true => {
          1 => OpenCL::Char1,
          2 => OpenCL::Short1,
          4 => OpenCL::Int1,
          8 => OpenCL::Long1
        },
        false => {
          1 => OpenCL::UChar1,
          2 => OpenCL::UShort1,
          4 => OpenCL::UInt1,
          8 => OpenCL::ULong1
        }
      }
    end

    def init_opencl(options)
      require 'opencl_ruby_ffi'
      init_opencl_types
      device = select_cl_device(options)
      @context = OpenCL::create_context([device])
      program = @context.create_program_with_source([@code.string])
      opts = options[:CLFLAGS]
      begin
        program.build(:options => options[:CLFLAGS])
      rescue OpenCL::Error => e
        puts e.to_s
        puts program.build_status
        puts program.build_log
        if options[:verbose] or get_verbose then
          puts @code.string
        end
        raise "OpenCL Failed to build #{@procedure.name}"
      end
      if options[:verbose] or get_verbose then
        program.build_log.each {|dev,log|
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
        param = @context.create_buffer( arg.size * arg.element_size, :flags => flags )
        @queue.enqueue_write_buffer( param, arg, :blocking => true )
      end
      return param
    end

    def create_opencl_scalar(arg, parameter)
      if parameter.type.is_a?(Real) then
        return @@opencl_real_types[parameter.type.size]::new(arg)
      elsif parameter.type.is_a?(Int) then
        return @@opencl_int_types[parameter.type.signed][parameter.type.size]::new(arg)
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
  if not gws then
    raise ":global_work_size or :block_number are required to run OpenCL kernels!" unless opts[:block_number]
    gws = []
    opts[:block_number].each_index { |i|
      raise "if using :block_number, :block_size is required  to run OpenCL kernels!" unless opts[:block_size]
      gws.push(opts[:block_number][i]*opts[:block_size][i])
    }
  end
  lws = opts[:local_work_size]
  if not lws then
    lws = opts[:block_size]
  end
  event = @queue.enqueue_NDrange_kernel(@kernel, gws, :local_work_size => lws)
  @procedure.parameters.each_index { |i|
    if @procedure.parameters[i].dimension and (@procedure.parameters[i].direction == :inout or @procedure.parameters[i].direction == :out) then
      read_opencl_param( params[i], args[i], @procedure.parameters[i] )
    end
  }
  @queue.finish
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

  end

end
