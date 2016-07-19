module BOAST

  # @!parse module Functors; functorize For; end
  class For < ControlStructure
    include Annotation
    ANNOTATIONS = [ :iterator, :start, :stop, :step, :operator ]

    attr_reader :iterator
    attr_reader :start
    attr_reader :stop
    attr_reader :step
    attr_accessor :block

    def unroll?
      return !!@unroll
    end

    def unroll=(val)
      @unroll = val
    end

    def initialize(iterator, start, stop, options={}, &block)
      default_options = {:step => 1}
      default_options.update( options )
      @options = options
      @iterator = iterator
      @start = start
      @stop = stop
      @step = default_options[:step]
      @operator = "<="
      @block = block
      @openmp = default_options[:openmp]
      @unroll = default_options[:unroll]
      @args = default_options[:args]
      if @openmp then
        if @openmp.kind_of?(Hash) then
          @openmp = OpenMP::For(@openmp)
        else
          @openmp = OpenMP::For({})
        end
      end
      begin
        push_env( :replace_constants => true )
        if @step.kind_of?(Variable) then
          step = @step.constant
        elsif @step.kind_of?(Expression) then
          step = eval "#{@step}"
        else
          step = @step.to_i
        end
        @operator = ">=" if step < 0
      rescue
        STDERR.puts "Warning could not determine sign of step (#{@step}) assuming positive" if [C, CL, CUDA].include?( lang ) and debug?
      ensure
        pop_env( :replace_constants )
      end
    end

    def get_c_strings
      return { :for => '"for (#{i} = #{b}; #{i} #{o} #{e}; #{i} += #{s}) {"',
               :end => '"}"' }
    end

    def get_fortran_strings
      return { :for => '"do #{i} = #{b}, #{e}, #{s}"',
               :end => '"end do"' }
    end

    private :get_c_strings, :get_fortran_strings

    alias get_cl_strings get_c_strings
    alias get_cuda_strings get_c_strings

    eval token_string_generator( * %w{for i b e s o})
    eval token_string_generator( * %w{end})

    def to_s
      s = for_string(@iterator, @start, @stop, @step, @operator)
      return s
    end

    def unroll
      opts = @options.clone
      opts[:unroll] = true
      return For::new(@iterator, @start, @stop, opts, &block)
    end

    def pr_unroll(*args)
      raise "Block not given!" if not @block
      begin
        begin
          push_env( :replace_constants => true )
          if @start.kind_of?(Variable) then
            start = @start.constant
          elsif @start.kind_of?(Expression) then
            start = eval "#{@start}"
          else
            start = @start.to_i
          end
          if @stop.kind_of?(Variable) then
            stop = @stop.constant
          elsif @stop.kind_of?(Expression) then
            stop = eval "#{@stop}"
          else
            stop = @stop.to_i
          end
          if @step.kind_of?(Variable) then
            step = @step.constant
          elsif @step.kind_of?(Expression) then
            step = eval "#{@step}"
          else
            step = @step.to_i
          end
          raise "Invalid bounds (not constants)!" if not ( start and stop and step )
        ensure
          pop_env( :replace_constants )
        end
      rescue Exception => ex
        open
        if @block then
          @block.call(*args)
          close
        end
        return self
      end
      range = start..stop
      @iterator.force_replace_constant = true
      range.step(step) { |i|
        @iterator.constant = i
        @block.call(*args)
      }
      @iterator.force_replace_constant = false
      @iterator.constant = nil
    end

    private :pr_unroll

    def open
      @openmp.open if @openmp
      s=""
      s += indent
      s += to_s
      output.puts s
      increment_indent_level      
      return self
    end 

    def pr(*args)
      args = @args if args.length == 0 and @args
      return pr_unroll(*args) if unroll?
      open
      if @block then
        @block.call(*args)
        close
      end
      return self
    end

    def close
      decrement_indent_level      
      s = ""
      s += indent
      s += end_string
      output.puts s
      @openmp.close if @openmp
      return self
    end

#    def u(s = 2)
#      return [For::new(@iterator, @start, @stop - (@step*s - 1), @options.dup.update( { :step => (@step*s) } ), &@block),
#              For::new(@iterator, @start.to_var + ((@stop - @start + 1)/(@step*s))*(@step*s), @stop, @options, &@block) ]
#    end

  end

end
