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

    def to_s(condition_number = 0)
      return self.to_s_fortran(condition_number) if BOAST::get_lang == FORTRAN
      return self.to_s_c(condition_number) if [C, CL, CUDA].include?( BOAST::get_lang )
    end

    def to_s_fortran(condition_number)
      s = ""
      if condition_number == 0 then
        s += "if ( #{@conditions.first} ) then"
      else
        if @conditions[condition_number] then
          s += "else if ( #{@conditions[condition_number]} ) then"
        else
          s += "else"
        end
      end
      return s
    end

    def to_s_c(condition_number)
      s = ""
      if condition_number == 0 then
        s += "if(#{@conditions.first}){"
      else
        if @conditions[condition_number] then
          s += "} else if(#{@conditions[condition_number]}){"
        else
          s += "} else {"
        end
      end
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
      if @blocks.size > 0 then
        BOAST::increment_indent_level
        @blocks.each_index { |indx|
          BOAST::decrement_indent_level
          s=""
          s += " "*BOAST::get_indent_level
          s += self.to_s(indx)
          BOAST::get_output.puts s
          BOAST::increment_indent_level
          @blocks[indx].call(*args)
        }
        self.close
      else
        self.decl
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
      s += "end if"
      BOAST::get_output.puts s
      return self
    end

  end

end
