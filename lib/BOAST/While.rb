module BOAST

  class While < BOAST::ControlStructure
    include BOAST::Inspectable
    extend BOAST::Functor

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
      BOAST::C => @@c_strings,
      BOAST::CL => @@c_strings,
      BOAST::CUDA => @@c_strings,
      BOAST::FORTRAN => @@f_strings
    }

    eval token_string_generator( * %w{while cond} )
    eval token_string_generator( * %w{end} )

    def to_s
      return while_string(@condition)
    end

    def open
      s=""
      s += BOAST::indent
      s += to_s
      BOAST::output.puts s
      BOAST::increment_indent_level
      return self
    end

    def print(*args)
      open
      if @block then
        @block.call(*args)
        close
      end
      return self
    end

    def close
      BOAST::decrement_indent_level      
      s = ""
      s += BOAST::indent
      s += end_string
      BOAST::output.puts s
      return self
    end

  end

end
