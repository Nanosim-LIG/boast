module BOAST

  class Case
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    attr_reader :expression
    attr_reader :constants_list

    def initialize(expression, *control)
      @expression = expression
      @constants_list = []
      @blocks = []
      if control.size < 1 then
        raise "No block given!"
      elsif control.size.even? then
        (0..control.size-1).step(2) { |i|
          @constants_list[i/2] = [control[i]].flatten
          @blocks[i/2] = control[i+1]
        }
      else
        (0..control.size-2).step(2) { |i|
          @constants_list[i/2] = [control[i]].flatten
          @blocks[i/2] = control[i+1]
        }
        @blocks.push(control.last)
      end
    end

    def to_s(constants, first= true)
      return self.to_s_fortran(constants, first) if BOAST::get_lang == FORTRAN
      return self.to_s_c(constants, first) if [C, CL, CUDA].include?( BOAST::get_lang )
    end

    def to_s_fortran(constants, first)
      s = ""
      if first then
        s += " "*BOAST::get_indent_level
        s += "select case (#{@expression})\n"
        BOAST::increment_indent_level
      else
        BOAST::decrement_indent_level
      end
      s += " "*BOAST::get_indent_level
      if constants and constants.size>0 then
        s += "case (#{constants.join(" : ")})"
      else
        s += "case default"
      end
      BOAST::increment_indent_level
      return s
    end

    def to_s_c(constants, first)
      s = ""
      if first then
        s += " "*BOAST::get_indent_level
        s += "switch(#{@expression}){\n"
        BOAST::increment_indent_level
      else
        s += " "*BOAST::get_indent_level + "break;\n"
        BOAST::decrement_indent_level
      end
      s += " "*BOAST::get_indent_level
      if constants and constants.size>0 then
        s += "case #{constants.join(" : case")} :"
      else
        s += "default :"
      end
      BOAST::increment_indent_level
      return s
    end

    def print(*args)
      first = true
      @blocks.each_index { |indx|
        s = self.to_s(@constants_list[indx],first)
        BOAST::get_output.puts s
        @blocks[indx].call(*args)
        first = false
      }
      self.close
      return self
    end
    def close(final=true)
      return self.close_fortran(final) if BOAST::get_lang == FORTRAN
      return self.close_c(final) if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def close_c(final=true)
      s = ""
      s += " "*BOAST::get_indent_level if final
      s += "break;\n"
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "}"
      BOAST::decrement_indent_level      
      BOAST::get_output.puts s if final
      return s
    end
    def close_fortran(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "end select"
      BOAST::decrement_indent_level      
      BOAST::get_output.puts s if final
      return s
    end

  end 

end
