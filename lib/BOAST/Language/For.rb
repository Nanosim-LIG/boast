module BOAST

  # @!parse module Functors; functorize For; end
  class For < ControlStructure
    include Annotation
    ANNOTATIONS = [ :iterator, :first, :last, :step, :operator ]

    attr_reader :iterator
    attr_reader :first
    attr_reader :last
    attr_reader :step
    attr_accessor :block

    # returns the Boolean evaluation of the unroll attribute.
    def unroll?
      return @unroll
    end

    # Sets the unroll attribute to val.
    def unroll=(val)
      @unroll = val
    end

    # Creates a new instance of the For construct.
    # @param [Variable] iterator
    # @param [#to_s] first iteration start
    # @param [#to_s] last iteration stop (inclusive)
    # @param [Hash] options contains named options
    # @param [Proc,nil] block if given, will be evaluated when {pr} is called
    # @option options [#to_s] :step spcifies the increment in the for loop
    # @option options [Boolean,Hash] :openmp specifies if an OpenMP For pragma has to be generated. If a Hash is specified it conatins the OpenMP clauses and their values.
    # @option options [Boolean, Integer] :unroll specifies if {pr} must try to unroll the loop
    # @option options [Array<Object>] :args arguments to be passed to the block. Will be superseded by those provided by {pr}
    def initialize(iterator, first, last, options={}, &block)
      super()
      default_options = {:step => 1}
      default_options.update( options )
      @options = options
      @iterator = iterator
      @first = first
      @last = last
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
      push_env( :replace_constants => true ) {
        begin
          if @step.kind_of?(Variable) then
            step = @step.constant
          elsif @step.kind_of?(Expression) then
            step = eval "#{@step}"
          else
            step = @step.to_i
          end
          @operator = ">=" if step < 0
        rescue
          $stderr.puts "Warning could not determine sign of step (#{@step}) assuming positive" if [C, CL, CUDA].include?( lang ) and debug?
        end
      }
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

    # Returns a string representation of the For construct.
    def to_s
      s = for_string(@iterator, @first, @last, @step, @operator)
      return s
    end

    # Creates a copy of this For construct with the unroll option set and returns it if it is different from the current unroll flag.
    # @return [For]
    # @param [Boolean, Integer] flag specifying if the For should be unrolled or not
    def unroll( flag = true )
      if flag != @unroll then
        opts = @options.clone
        opts[:unroll] = flag
        return For::new(@iterator, @first, @last, opts, &block)
      else
        return self
      end
    end

    def pr_unroll_true(*args, &block)
      begin
        begin
          push_env( :replace_constants => true )
          if @first.kind_of?(Variable) then
            first = @first.constant
          elsif @first.kind_of?(Expression) then
            first = eval "#{@first}"
          else
            first = @first.to_i
          end
          if @last.kind_of?(Variable) then
            last = @last.constant
          elsif @last.kind_of?(Expression) then
            last = eval "#{@last}"
          else
            last = @last.to_i
          end
          if @step.kind_of?(Variable) then
            step = @step.constant
          elsif @step.kind_of?(Expression) then
            step = eval "#{@step}"
          else
            step = @step.to_i
          end
          raise "Invalid bounds (not constants)!" unless ( first and last and step )
        ensure
          pop_env( :replace_constants )
        end
      rescue Exception
        open
        block.call(*args)
        close
        return self
      end
      range = first..last
      @iterator.force_replace_constant = true
      range.step(step) { |i|
        @iterator.constant = i
        block.call(*args)
      }
      @iterator.force_replace_constant = false
      @iterator.constant = nil
    end

    def pr_unroll_integer(*args, &block)
      raise "Invalid unroll factor: #{@unroll}!" if @unroll < 1
      old_step = @step
      old_last = @last
      @last = @last - (@unroll-1) * @step
      @step = @step * @unroll
      open
      @step = old_step
      @last = old_last
      it = "#{@iterator}"
      @iterator.force_replace_constant = true
      @unroll.times { |k|
        @iterator.constant = Int(it) + k * @step
        block.call(*args)
      }
      @iterator.force_replace_constant = false
      @iterator.constant = nil
      close
      old_first = @first
      @first = @first + ((@last + (1 - @first))/(@step * @unroll))*( @step * @unroll )
      open
      @first = old_first
      block.call(*args)
      close
    end

    def pr_unroll(*args, &block)
      block = @block unless block
      raise "Block not given!" unless block
      case @unroll
      when TrueClass
        pr_unroll_true(*args, &block)
      when Integer
        pr_unroll_integer(*args, &block)
      else
        raise "Invalid unroll factor: #{@unroll.inspect}!"
      end
    end

    private :pr_unroll

    # Opens the For construct (keyword, iterator, bounds, step, opening bracket in C like languages). The result is printed to the BOAST output.
    # @return [self]
    def open
      @openmp.open if @openmp
      s=""
      s << indent
      s << to_s
      output.puts s
      increment_indent_level      
      return self
    end 

    # Prints the For construct to the BOAST output (see {open}).
    # If a block is provided during initialization, it will be printed and the construct will be closed (see {close}).
    # @param [Array<Object>] args any number of arguments to pass to the block
    # @param [Proc] block an optional block to be evaluated. Supersede the one given at initialization
    # @return [self]
    def pr(*args, &block)
      args = @args if args.length == 0 and @args
      block = @block unless block
      return pr_unroll(*args, &block) if unroll?
      open
      if block then
        block.call(*args)
        close
      end
      return self
    end

    # Closes the For construct (keyword, closing bracket in C like languages). The result is printed to the BOAST output.
    # @return [self]
    def close
      decrement_indent_level      
      s = ""
      s << indent
      s << end_string
      output.puts s
      @openmp.close if @openmp
      return self
    end

#    def u(s = 2)
#      return [For::new(@iterator, @first, @last - (@step*s - 1), @options.dup.update( { :step => (@step*s) } ), &@block),
#              For::new(@iterator, @first.to_var + ((@last - @first + 1)/(@step*s))*(@step*s), @last, @options, &@block) ]
#    end

  end

end
