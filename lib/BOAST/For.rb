module BOAST

  class For < ControlStructure

    attr_reader :iterator
    attr_reader :begin
    attr_reader :end
    attr_reader :step

    def initialize(i, b, e, options={}, &block)
      default_options = {:step => 1}
      default_options.update( options )
      @iterator = i
      @begin = b
      @end = e
      @step = default_options[:step]
      @operator = "<="
      @block = block
      @openmp = default_options[:openmp]
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

    @@c_strings = {
      :for => '"for (#{i} = #{b}; #{i} #{o} #{e}; #{i} += #{s}) {"',
      :end => '"}"',
      :openmp_for => '"#pragma omp for #{c}"',
      :openmp_end => '""'
    }

    @@f_strings = {
      :for => '"do #{i} = #{b}, #{e}, #{s}"',
      :end => '"end do"',
      :openmp_for => '"!$omp do #{c}"',
      :openmp_end => '"!$omp end do #{c}"'
    }

    @@strings = {
      C => @@c_strings,
      CL => @@c_strings,
      CUDA => @@c_strings,
      FORTRAN => @@f_strings
    }

    eval token_string_generator( * %w{for i b e s o})
    eval token_string_generator( * %w{end})

    def to_s
      s = for_string(@iterator, @begin, @end, @step, @operator)
      return s
    end

    def unroll(*args)
      raise "Block not given!" if not @block
      push_env( :replace_constants => true )
      begin
        if @begin.kind_of?(Variable) then
          start = @begin.constant
        elsif @begin.kind_of?(Expression) then
          start = eval "#{@begin}"
        else
          start = @begin.to_i
        end
        if @end.kind_of?(Variable) then
          e = @end.constant
        elsif @end.kind_of?(Expression) then
          e = eval "#{@end}"
        else
          e = @end.to_i
        end
        if @step.kind_of?(Variable) then
          step = @step.constant
        elsif @step.kind_of?(Expression) then
          step = eval "#{@step}"
        else
          step = @step.to_i
        end
        raise "Invalid bounds (not constants)!" if not ( start and e and step )
      rescue Exception => ex
        if not ( start and e and step ) then
          pop_env( :replace_constants )
          return pr(*args) if not ( start and e and step )
        end
      end
      pop_env( :replace_constants )
      range = start..e
      @iterator.force_replace_constant = true
      range.step(step) { |i|
        @iterator.constant = i
        @block.call(*args)
      }
      @iterator.force_replace_constant = false
      @iterator.constant = nil
    end

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

  end

end
