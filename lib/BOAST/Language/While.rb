module BOAST

  class While < ControlStructure

    attr_reader :condition

    def initialize(condition, &block)
      @condition = condition
      @block = block
    end

    def get_c_strings
      return { :while => '"while (#{cond}) {"',
               :end => '"}"' }
    end

    def get_fortran_strings
      return { :while => '"do while (#{cond})"',
               :end => '"end do"' }
    end

    alias get_cl_strings get_c_strings
    alias get_cuda_strings get_c_strings

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
