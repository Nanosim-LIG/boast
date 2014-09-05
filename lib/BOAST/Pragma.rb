module BOAST

  class Pragma
    include BOAST::Inspectable
    extend BOAST::Functor

    attr_reader :name
    attr_reader :options

    def initialize(name, options)
      @name = name
      @options = options
    end

    def to_s
      s = ""
      if BOAST::get_lang == FORTRAN then
        s += "$!"
      else
        s += "#pragma"
      end
      @options.each{ |opt|
        s += " #{opt}"
      }
      return s
    end

    def print
      s=""
      s += to_s
      BOAST::get_output.puts s
      return self
    end
  end

end
