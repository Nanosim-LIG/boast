module BOAST

  class If
    include BOAST::Inspectable
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    attr_reader :conditions
    def initialize(*conditions, &block)
      @conditions = []
      @blocks = []
      if conditions.size == 0 then
        raise "Illegal if construct!"
      elsif conditions.size == 1 then
        @conditions.push(conditions[0])
        @blocks.push(block)
      elsif conditions.size.even? then
        (0..conditions.size-1).step(2) { |i|
          @conditions[i/2] = conditions[i]
          @blocks[i/2] = conditions[i+1]
        }
      else
        (0..conditions.size-2).step(2) { |i|
          @conditions[i/2] = conditions[i]
          @blocks[i/2] = conditions[i+1]
        }
        @blocks.push(conditions.last)
      end
    end

    def to_s(condition, first= true)
      return self.to_s_fortran(condition, first) if BOAST::get_lang == FORTRAN
      return self.to_s_c(condition, first) if [C, CL, CUDA].include?( BOAST::get_lang )
    end

    def to_s_fortran(condition, first)
      s = ""
      if first then
        s += "if ( #{condition} ) then"
      else
        if condition then
          s += "else if ( #{condition} ) then"
        else
          s += "else"
        end
      end
      return s
    end

    def to_s_c(condition, first)
      s = ""
      if first then
        s += "if(#{condition}){"
      else
        if condition then
          s += "} else if(#{condition}){"
        else
          s += "} else {"
        end
      end
      return s
    end

    def print(*args)
      s=""
      s += " "*BOAST::get_indent_level
      s += self.to_s(@conditions.first)
      BOAST::increment_indent_level      
      BOAST::get_output.puts s
      if @blocks.size > 0 then
        if @blocks[0] then
          @blocks[0].call(*args)
        end
        @blocks[1..-1].each_index { |indx|
          BOAST::decrement_indent_level      
          s=""
          s += " "*BOAST::get_indent_level 
          s += self.to_s(@conditions[1..-1][indx],false)
          BOAST::increment_indent_level
          BOAST::get_output.puts s
          @blocks[1..-1][indx].call(*args)
        }
        self.close
      end
      return self
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
      s += "end if"
      BOAST::get_output.puts s if final
      return s
    end

  end

end 
