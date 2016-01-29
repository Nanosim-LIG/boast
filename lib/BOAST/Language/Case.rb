module BOAST

  class Case < ControlStructure

    attr_reader :expression
    attr_reader :constants_list

    def initialize(expression, *control, &block)
      @expression = expression
      @constants_list = []
      @blocks = []
      control.push(block) if block
      if control.size < 1 then
        raise "No block given!"
      else
        while control.size >= 2 do
          @constants_list.push [control.shift].flatten
          @blocks.push control.shift
        end
        @blocks.push control.shift if control.size > 0
      end
    end

    def get_c_strings
      return { :switch => '"switch (#{expr}) {"',
               :case => '"case #{constants.join(" : case ")} :"',
               :default => '"default :"',
               :break => '"break;"',
               :end => '"}"' }
    end

    def get_fortran_strings
      return { :switch => '"select case (#{expr})"',
               :case => '"case (#{constants.join(", ")})"',
               :default => '"case default"',
               :break => 'nil',
               :end => '"end select"' }
    end

    alias get_cl_strings get_c_strings
    alias get_cuda_strings get_c_strings

    eval token_string_generator( * %w{switch expr})
    eval token_string_generator( * %w{case constants})
    eval token_string_generator( * %w{default})
    eval token_string_generator( * %w{break})
    eval token_string_generator( * %w{end})

    def to_s
      s = ""
      s += switch_string(@expression)
      return s
    end

    def pr_block(block_number, *args)
      c = @constants_list[block_number] and @constants_list[block_number].size > 0
      if c then
        output.puts indent + case_string(@constants_list[block_number])
      else
        output.puts indent + default_string
      end
      increment_indent_level
      @blocks[block_number].call(*args)
      if c then
        output.puts indent + break_string + "\n" if break_string
      end
      decrement_indent_level
    end

    def open
      s = ""
      s += indent
      s += to_s
      output.puts s
      increment_indent_level
      return self
    end

    def pr(*args)
      open
      if @blocks.size > 0 then
        @blocks.each_index { |indx|
          pr_block(indx, *args)
        }
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
