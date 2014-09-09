module BOAST

  class Case < BOAST::ControlStructure
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

    @@c_strings = {
      :switch => '"switch (#{expr}) {"',
      :case => '"case #{constants.join(" : case")} :"',
      :default => '"default :"',
      :break => '"break;"',
      :end => '"}"'
    }

    @@f_strings = {
      :switch => '"select case (#{expr})"',
      :case => '"case (#{constants.join(" : ")})"',
      :default => '"case default"',
      :break => 'nil',
      :end => '"end select"'
    }

    @@strings = {
      BOAST::C => @@c_strings,
      BOAST::CL => @@c_strings,
      BOAST::CUDA => @@c_strings,
      BOAST::FORTRAN => @@f_strings
    }

    eval token_string_generator( * %w{switch expr})
    eval token_string_generator( * %w{case constants})
    eval token_string_generator( * %w{default})
    eval token_string_generator( * %w{break})
    eval token_string_generator( * %w{end})

    def to_s(block_number = nil)
      s = ""
      if block_number then
        if block_number != 0 then
          s += BOAST::indent + break_string + "\n" if break_string
          BOAST::decrement_indent_level
        end
        s += BOAST::indent
        if @constants_list[block_number] and @constants_list[block_number].size > 0 then
          s += case_string(@constants_list[block_number])
        else
          s += default_string
        end
      else
        s += BOAST::indent
        s += switch_string(@expression)
      end
      BOAST::increment_indent_level
      return s
    end

    def open
      BOAST::output.puts to_s
      return self
    end

    def pr(*args)
      open
      if @blocks.size > 0 then
        @blocks.each_index { |indx|
          s = to_s(indx)
          BOAST::output.puts s
          @blocks[indx].call(*args)
        }
        close
      end
      return self
    end

    def close
      s = ""
      s += BOAST::indent + break_string + "\n" if break_string
      BOAST::decrement_indent_level      
      s += BOAST::indent
      s += end_string
      BOAST::decrement_indent_level      
      BOAST::output.puts s
      return self
    end

  end 

end
