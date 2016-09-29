module BOAST

  # @!parse module Functors; functorize While; end
  class While < ControlStructure
    include Annotation
    ANNOTATIONS = [ :condition ]

    attr_reader :condition

    # Creates a new instance of the While construct.
    # @param [Expression] condition
    # @param [Proc,nil] block if given, will be evaluated when {pr} is called
    def initialize(condition, &block)
      @condition = condition
      @block = block
    end

    def get_c_strings
      return { :while => '"while (#{cond}) {"',
               :end => '"}"' }
    end

    def get_fortran_strings
      return { :while => '"do while (#{cond})"',
               :end => '"end do"' }
    end

    private :get_c_strings, :get_fortran_strings

    alias get_cl_strings get_c_strings
    alias get_cuda_strings get_c_strings

    eval token_string_generator( * %w{while cond} )
    eval token_string_generator( * %w{end} )

    # Returns a string representation of the While construct.
    def to_s
      return while_string(@condition)
    end

    # Opens the While construct (keyword, condition, opening bracket in C like languages). The result is printed to the BOAST output.
    # @return [self]
    def open
      s=""
      s += indent
      s += to_s
      output.puts s
      increment_indent_level
      return self
    end

    # Prints the While construct to the BOAST output (see {open}).
    # If a block is provided during initialization, it will be printed and the construct will be closed (see {close}).
    # @param [Array<Object>] args any number of arguments to pass to the block
    # @param [Proc] block an optional block to be evaluated. Supersede the one given at initialization
    # @return [self]
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

    # Closes the While construct (keyword, closing bracket in C like languages). The result is printed to the BOAST output.
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
