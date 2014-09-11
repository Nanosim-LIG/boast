module BOAST

  class While < ControlStructure

    attr_reader :condition

    def initialize(condition, &block)
      @condition = condition
      @block = block
    end

    @@c_strings = {
      :while => '"while (#{cond}) {"',
      :end => '"}"'
    }

    @@f_strings = {
      :while => '"do while (#{cond})"',
      :end => '"end do"'
    }

    @@strings = {
      C => @@c_strings,
      CL => @@c_strings,
      CUDA => @@c_strings,
      FORTRAN => @@f_strings
    }

    eval token_string_generator( * %w{while cond} )
    eval token_string_generator( * %w{end} )

    def to_s
      return while_string(@condition)
    end

    def open
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
      return self
    end

  end

end
