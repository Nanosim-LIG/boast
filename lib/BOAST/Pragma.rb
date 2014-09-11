module BOAST

  class Pragma
    include PrivateStateAccessor
    include Inspectable
    extend Functor

    attr_reader :name
    attr_reader :options

    def initialize(name, options)
      @name = name
      @options = options
    end

    def to_s
      s = ""
      if lang == FORTRAN then
        s += "$!"
      else
        s += "#pragma"
      end
      @options.each{ |opt|
        s += " #{opt}"
      }
      return s
    end

    def pr
      s=""
      s += to_s
      output.puts s
      return self
    end
  end

end
