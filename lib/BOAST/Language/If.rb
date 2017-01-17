module BOAST

  # @!parse module Functors; functorize If; end
  class If < ControlStructure

    attr_reader :conditions

    # Creates a new instance of the If construct
    # @overload initialize(condition, &block)
    #   Creates a simple If construct
    #   @param [Expression] condition
    #   @param [Proc,nil] block if given, will be evaluated when {pr} is called
    # @overload initialize(conditions, &block)
    #   Creates a multi-condition If construct
    #   @param [Hash{Expression, :else => Proc}] conditions each condition and its associated block (can be nil)
    #   @param [Proc,nil] block else block if :else is not specified in the conditions or nil
    def initialize(conditions, &block)
      super()
      @conditions = []
      @blocks = []
      if conditions.is_a?(Hash) then
        else_block = conditions.delete(:else)
        else_block = block unless else_block or not block
        conditions.each { |key, value|
          @conditions.push key
          @blocks.push value
        }
        @blocks.push else_block if else_block
      else
        @conditions.push conditions
        @blocks.push block if block
      end
    end

    def get_c_strings
      return { :if => '"if (#{cond}) {"',
               :elsif => '"} else if (#{cond}) {"',
               :else => '"} else {"',
               :end => '"}"' }
    end

    def get_fortran_strings
      return { :if => '"if (#{cond}) then"',
               :elsif => '"else if (#{cond}) then"',
               :else => '"else"',
               :end => '"end if"' }
    end

    private :get_c_strings, :get_fortran_strings

    alias get_cl_strings get_c_strings
    alias get_cuda_strings get_c_strings

    eval token_string_generator( * %w{if cond} )
    eval token_string_generator( * %w{elsif cond} )
    eval token_string_generator( * %w{else} )
    eval token_string_generator( * %w{end} )

    # Returns a string representation of the If construct.
    # @param [Fixnum] condition_number condition to print
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

    # Opens the If construct. The result is printed on the BOAST output. If a condition number is given, will print the corresponding condition (or else if none exist)
    # @param [Fixnum] condition_number condition to print
    # @return [self]
    def open(condition_number = 0)
      decrement_indent_level if condition_number > 0
      s = ""
      s += indent
      s += to_s(condition_number)
      output.puts s
      increment_indent_level
      return self
    end

    # Prints the If construct to the BOAST output (see {open}).
    # If block/blocks is/are provided during initialization, they will be printed and the construct will be closed (see {close}).
    # @param [Array<Object>] args any number of arguments to pass to the block/blocks
    # @return [self]
    def pr(*args)
      args = @args if args.length == 0 and @args
      if @blocks.size > 0 then
        @blocks.each_index { |indx|
          open(indx)
          @blocks[indx].call(*args)
        }
        close
      else
        open
      end
      return self
    end

    # Closes the If construct (keyword, closing bracket in C like languages). The result is printed to the BOAST output.
    # @return [self]
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
