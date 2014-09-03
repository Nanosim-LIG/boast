module BOAST

  class Case
    include BOAST::Inspectable
    extend BOAST::Functor

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

    def to_s(block_number = nil)
      return self.to_s_fortran(block_number) if BOAST::get_lang == FORTRAN
      return self.to_s_c(block_number) if [C, CL, CUDA].include?( BOAST::get_lang )
    end

    def to_s_fortran(block_number)
      s = ""
      if block_number then
        BOAST::decrement_indent_level if block_number != 0
        s += " "*BOAST::get_indent_level
        if @constants_list[block_number] and @constants_list[block_number].size > 0 then
          s += "case (#{@constants_list[block_number].join(" : ")})"
        else
          s += "case default"
        end
      else
        s += " "*BOAST::get_indent_level
        s += "select case (#{@expression})"
      end
      BOAST::increment_indent_level
      return s
    end

    def to_s_c(block_number)
      s = ""
      if block_number then
        if block_number != 0 then
          s += " "*BOAST::get_indent_level + "break;\n"
          BOAST::decrement_indent_level
        end
        s += " "*BOAST::get_indent_level
        if @constants_list[block_number] and @constants_list[block_number].size > 0 then
          s += "case #{@constants_list[block_number].join(" : case")} :"
        else
          s += "default :"
        end
      else
        s += " "*BOAST::get_indent_level
        s += "switch(#{@expression}){"
      end
      BOAST::increment_indent_level
      return s
    end

    def decl
      BOAST::get_output.puts self.to_s
      return self
    end

    def print(*args)
      self.decl
      if @blocks.size > 0 then
        @blocks.each_index { |indx|
          s = self.to_s(indx)
          BOAST::get_output.puts s
          @blocks[indx].call(*args)
        }
        self.close
      end
      return self
    end

    def close
      return self.close_fortran if BOAST::get_lang == FORTRAN
      return self.close_c if [C, CL, CUDA].include?( BOAST::get_lang )
    end

    def close_c
      s = ""
      s += " "*BOAST::get_indent_level
      s += "break;\n"
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level
      s += "}"
      BOAST::decrement_indent_level      
      BOAST::get_output.puts s
      return self
    end

    def close_fortran
      BOAST::decrement_indent_level      
      s = ""
      s += " "*BOAST::get_indent_level
      s += "end select"
      BOAST::decrement_indent_level      
      BOAST::get_output.puts s
      return self
    end

  end 

end
