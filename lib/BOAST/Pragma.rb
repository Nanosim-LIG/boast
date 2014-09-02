module BOAST

  class Pragma
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

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

    def print(final = true)
      s=""
      s += self.to_s
      BOAST::get_output.puts s if final
      return s
    end
  end

end
