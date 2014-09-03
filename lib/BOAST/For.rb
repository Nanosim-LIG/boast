module BOAST

  class For < ControlStructure
    include BOAST::Inspectable
    extend BOAST::Functor

    attr_reader :iterator
    attr_reader :begin
    attr_reader :end
    attr_reader :step

    def initialize(i, b, e, s=1, &block)
      @iterator = i
      @begin = b
      @end = e
      @step = s
      @block = block
    end

    @@c_strings = {
      :for => '"for (#{i} = #{b}; #{i} <= #{e}; #{i} += #{s}) {"',
      :end => '"}"'
    }

    @@f_strings = {
      :for => '"do #{i} = #{b}, #{e}, #{s}"',
      :end => '"end do"'
    }

    @@strings = {
      BOAST::C => @@c_strings,
      BOAST::CL => @@c_strings,
      BOAST::CUDA => @@c_strings,
      BOAST::FORTRAN => @@f_strings
    }

    eval token_string_generator( * %w{for i b e s})
    eval token_string_generator( * %w{end})

    def to_s
      s = for_string(@iterator, @begin, @end, @step)
      return s
    end

    def unroll(*args)
      raise "Block not given!" if not @block
      BOAST::push_env( :replace_constants => true )
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
          BOAST::pop_env( :replace_constants )
          return self.print(*args) if not ( start and e and step )
        end
      end
      BOAST::pop_env( :replace_constants )
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
      s=""
      s += " "*BOAST::get_indent_level
      s += self.to_s
      BOAST::get_output.puts s
      BOAST::increment_indent_level      
      return self
    end 

    def print(*args)
      self.open
      if @block then
        @block.call(*args)
        self.close
      end
      return self
    end

    def close
      BOAST::decrement_indent_level      
      s = ""
      s += " "*BOAST::get_indent_level
      s += end_string
      BOAST::get_output.puts s
      return self
    end

  end

end
