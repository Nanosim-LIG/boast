module BOAST

  class While
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

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
    def print(*args)
      final = true
      s=""
      s += " "*BOAST::get_indent_level if final
      s += self.to_s
      BOAST::increment_indent_level      
      BOAST::get_output.puts s if final
      if @block then
        s += "\n"
        @block.call(*args)
        s += self.close
      end
      return s
    end
    def close(final=true)
      return self.close_fortran(final) if BOAST::get_lang == FORTRAN
      return self.close_c(final) if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def close_c(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "}"
      BOAST::get_output.puts s if final
      return s
    end
    def close_fortran(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "end do"
      BOAST::get_output.puts s if final
      return s
    end

  end

end
