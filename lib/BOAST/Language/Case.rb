module BOAST

  # @!parse module Functors; functorize CaseCondition; end
  class CaseCondition < ControlStructure
    attr_reader :block
    attr_reader :constants

    def initialize(constants = nil, &block)
      super()
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

    private :get_c_strings, :get_fortran_strings

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

    def pr(*args, &block)
      args = @args if args.length == 0 and @args
      block = @block unless block
      open
      if block then
        block.call(*args)
        close
      end
      return self
    end

  end

  # @!parse module Functors; functorize Case; end
  class Case < ControlStructure

    attr_reader :expression
    attr_reader :case_conditions

    # Creates a new instance of the Caonstruct
    # @param [#to_s] expression tested Expression/Variable
    # @param [Hash{#to_s, :default => Proc}] control conditions and associated blocks.
    # @param [Proc,nil] block if provided, and :default is not defined in control (or nil), will be used as the default block.
    def initialize(expression, control = {}, &block)
      super()
      @expression = expression
      @case_conditions = []
      default = control.delete(:default)
      default = block unless default or not block
      control.each { |key, value|
        @case_conditions.push CaseCondition::new( [key].flatten, &value )
      }
      @case_conditions.push CaseCondition::new( &default ) if default
    end

    def get_c_strings
      return { :switch => '"switch (#{expr}) {"',
               :end => '"}"' }
    end

    def get_fortran_strings
      return { :switch => '"select case (#{expr})"',
               :end => '"end select"' }
    end

    private :get_c_strings, :get_fortran_strings

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
      args = @args if args.length == 0 and @args
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
