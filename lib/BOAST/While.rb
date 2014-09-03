module BOAST

  class While
    include BOAST::Inspectable
    extend BOAST::Functor

    attr_reader :condition

    def initialize(condition, &block)
      @condition = condition
      @block = block
    end

    def to_s
      return self.to_s_fortran if BOAST::get_lang == FORTRAN
      return self.to_s_c if [C, CL, CUDA].include?( BOAST::get_lang )
    end

    def to_s_fortran
      s = ""
      s += "do while( #{@condition} )"
      return s
    end

    def to_s_c
      s = ""
      s += "while(#{@condition}){"
      return s
    end

    def decl
      s=""
      s += " "*BOAST::get_indent_level
      s += self.to_s
      BOAST::get_output.puts s
      BOAST::increment_indent_level
      return self
    end

    def print(*args)
      self.decl
      if @block then
        @block.call(*args)
        self.close
      end
      return self
    end

    def close
      return self.close_fortran if BOAST::get_lang == FORTRAN
      return self.close_c if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def close_c
      BOAST::decrement_indent_level      
      s = ""
      s += " "*BOAST::get_indent_level
      s += "}"
      BOAST::get_output.puts s
      return self
    end

    def close_fortran
      BOAST::decrement_indent_level
      s = ""
      s += " "*BOAST::get_indent_level
      s += "end do"
      BOAST::get_output.puts s
      return self
    end

  end

end
