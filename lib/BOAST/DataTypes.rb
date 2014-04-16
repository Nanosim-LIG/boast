module BOAST

  class Real
    def self.parens(*args,&block)
      return Variable::new(args[0], self, *args[1..-1], &block)
    end

    attr_reader :size
    def initialize(hash={})
      if hash[:size] then
        @size = hash[:size]
      else
        @size = BOAST::get_default_real_size
      end
      if hash[:vector_length] and hash[:vector_length] > 1 then
        @vector_length = hash[:vector_length]
      else
        @vector_length = 1
      end
    end
    def decl
      return "real(kind=#{@size})" if BOAST::get_lang == FORTRAN
      if [C, CL, CUDA].include?( BOAST::get_lang ) and @vector_length == 1 then
        return "float" if @size == 4
        return "double" if @size == 8
      elsif [CL, CUDA].include?(BOAST::get_lang) and @vector_length > 1 then
        return "float#{@vector_length}" if @size == 4
        return "double#{@vector_length}" if @size == 8
      end
    end
  end

  class Int
    def self.parens(*args,&block)
      return Variable::new(args[0], self, *args[1..-1], &block)
    end

    attr_reader :size
    attr_reader :signed
    def initialize(hash={})
      if hash[:size] then
        @size = hash[:size]
      else
        @size = BOAST::get_default_int_size
      end
      if hash[:signed] != nil then
        @signed = hash[:signed]
      else
        @signed = BOAST::get_default_int_signed
      end
    end
    def decl
      return "integer(kind=#{@size})" if BOAST::get_lang == FORTRAN
      return "int#{8*@size}_t" if BOAST::get_lang == C
      if BOAST::get_lang == CL then
        #char="cl_"
        char=""
        char += "u" if not @signed
        return char += "char" if @size==1
        return char += "short" if @size==2
        return char += "int" if @size==4
        return char += "long" if @size==8
      elsif BOAST::get_lang == CUDA then
        char = ""
        char += "unsigned " if not @signed
        return char += "char" if @size==1
        return char += "short" if @size==2
        return char += "int" if @size==4
        return char += "long long" if @size==8
      end
    end
  end

  class CStruct
    attr_reader :name, :members, :members_array
    def self.parens(*args,&block)
      return Variable::new(args[0], self, *args[1..-1], &block)
    end

    def initialize(hash={})
      @name = hash[:type_name]
      @members = {}
      @members_array = []
      hash[:members].each { |m|
        mc = m.copy
        @members_array.push(mc)
        @members[mc.name] = mc
      }
    end

    def decl
      return "struct #{@name}" if [C, CL, CUDA].include?( BOAST::get_lang )
      return "TYPE(#{@name})" if BOAST::get_lang == FORTRAN
    end

    def finalize
       s = ""
       s += ";" if [C, CL, CUDA].include?( BOAST::get_lang )
       s+="\n"
       return s
    end

    def indent
       return " "*BOAST::get_indent_level
    end

    def header
      return header_c if [C, CL, CUDA].include?( BOAST::get_lang )
      return header_fortran if BOAST::get_lang == FORTRAN
      raise "Unsupported language!"
    end

    def header_c(final = true)
      s = ""
      s += self.indent if final
      s += self.decl + " {\n"
      @members_array.each { |value|
         s+= self.indent if final
         s+= " "*BOAST::get_indent_increment + value.decl(false)+";\n"
      }
      s += self.indent if final
      s += "}"
      s += self.finalize if final
      BOAST::get_output.print s if final
      return s
    end
    
    def header_fortran(final = true)
      s = ""
      s += self.indent if final
      s += "TYPE :: #{@name}\n"
      members_array.each { |value|
         s+= self.indent if final
         s+= " "*BOAST::get_indent_increment + value.decl(false)+"\n"
      }
      s += self.indent if final
      s += "END TYPE #{@name}"
      s += self.finalize if final
      BOAST::get_output.print s if final
      return s
    end

  end

  class CustomType
    attr_reader :size, :name, :vector_length
    def initialize(hash={})
      @name = hash[:type_name]
      @size = hash[:size]
      @vector_length = hash[:vector_length]
    end
    def decl
      return "#{@name}" if [C, CL, CUDA].include?( BOAST::get_lang )
    end
  end

end
