module BOAST

  class If < ControlStructure

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

    def get_c_strings
      return { :if => '"if (#{cond}) {"',
               :else_if => '"} else if (#{cond}) {"',
               :else => '"} else {"',
               :end => '"}"' }
    end

    def get_fortran_strings
      return { :if => '"if (#{cond}) then"',
               :elsif => '"else if (#{cond}) then"',
               :else => '"else"',
               :end => '"end if"' }
    end

    alias get_cl_strings get_c_strings
    alias get_cuda_strings get_c_strings

    eval token_string_generator( * %w{if cond} )
    eval token_string_generator( * %w{elsif cond} )
    eval token_string_generator( * %w{else} )
    eval token_string_generator( * %w{end} )

    def to_s(condition_number = 0)
      s = ""
      if condition_number == 0 then
        s += if_string(@conditions.first)
      else
        if @conditions[condition_number] then
          s += elsif_string(@conditions[condition_number])
        else
          s += else_string
        end
      end
      return s
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
      if @blocks.size > 0 then
        increment_indent_level
        @blocks.each_index { |indx|
          decrement_indent_level
          s=""
          s += indent
          s += to_s(indx)
          output.puts s
          increment_indent_level
          @blocks[indx].call(*args)
        }
        close
      else
        open
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
