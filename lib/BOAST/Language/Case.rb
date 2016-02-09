module BOAST

  class CaseCondition < ControlStructure
    attr_reader :block
    attr_reader :constants

    def initialize(constants = nil, &block)
      @constants = constants
      @block = block
    end

    def get_c_strings
      return { :case => '"case #{constants.join(" : case ")} :"',
               :default => '"default :"',
               :break => '"break;"' }
    end

    def get_fortran_strings
      return { :case => '"case (#{constants.join(", ")})"',
               :default => '"case default"',
               :break => 'nil' }
    end

    alias get_cl_strings get_c_strings
    alias get_cuda_strings get_c_strings

    eval token_string_generator( * %w{case constants})
    eval token_string_generator( * %w{default})
    eval token_string_generator( * %w{break})

    def to_s
      s = ""
      if @constants then
        s += case_string(@constants)
      else
        s += default_string
      end
      return s
    end

    def open
      s = ""
      s += indent
      s += to_s
      output.puts s
      increment_indent_level
      return self
    end

    def close
      if @constants and break_string then
        s = ""
        s += indent
        s += break_string
        output.puts s
      end
      decrement_indent_level
      return self
    end

    def pr(*args)
      open
      if @block then
        @block.call(*args)
        close
      end
      return self
    end

  end

  class Case < ControlStructure

    attr_reader :expression
    attr_reader :case_conditions

    def initialize(expression, *control, &block)
      @expression = expression
      @case_conditions = []
      control.push(block) if block
      while control.size >= 2 do
        @case_conditions.push CaseCondition::new([control.shift].flatten, &(control.shift))
      end
      @case_conditions.push CaseCondition::new(&(control.shift)) if control.size > 0
    end

    def get_c_strings
      return { :switch => '"switch (#{expr}) {"',
               :end => '"}"' }
    end

    def get_fortran_strings
      return { :switch => '"select case (#{expr})"',
               :end => '"end select"' }
    end

    alias get_cl_strings get_c_strings
    alias get_cuda_strings get_c_strings

    eval token_string_generator( * %w{switch expr})
    eval token_string_generator( * %w{end})

    def to_s
      s = ""
      s += switch_string(@expression)
      return s
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
      if @case_conditions.size > 0 then
        @case_conditions.each { |cond|
          cond.pr(*args)
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
