module BOAST

  # @!parse module Functors; functorize FuncCall; end
  class FuncCall
    include PrivateStateAccessor
    include Arithmetic
    include Inspectable
    extend Functor

    attr_reader :func_name
    attr_reader :args
    attr_accessor :prefix

    def initialize(func_name, *args)
      @prefix = nil
      @func_name = func_name
      if args.last.kind_of?(Hash) then
        @options = args.last
        @args = args[0..-2]
      else
        @args = args
        @options = {}
      end
      @return_type = @options[:return] ? @options[:return] : @options[:returns] if @options
    end

    def type
      return @return_type.type if @return_type
    end

    def to_var
      if @return_type then
        if @return_type.kind_of?(Variable)
          return @return_type.copy("#{self}")
        else
          return Variable::new("#{self}", @return_type)
        end
      else
        return Variable::new("#{self}", get_default_type)
      end
    end
      
    def to_s
      return to_s_fortran if lang == FORTRAN
      return to_s_c if CLANGS.include?( lang )
    end

    def to_s_fortran
      s = ""
      s << @prefix if @prefix
      s << "#{func_name}(#{@args.collect(&:to_s).join(", ")})"
    end

    def to_s_c
      s = ""
      s << @prefix if @prefix
      s << "#{func_name}(#{@args.collect(&:to_s).join(", ")})"
    end

    private :to_s_fortran, :to_s_c

    def pr
      s=""
      s << indent
      s << to_s
      s << ";" if CLANGS.include?( lang )
      output.puts s
      return self
    end
  end

  module Functors
    alias Call FuncCall
  end

end
