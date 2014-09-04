module BOAST

  class Sizet
    extend BOAST::VarFunctor

    attr_reader :signed
    attr_reader :size
    attr_reader :vector_length
    def initialize(hash={})
      if hash[:signed] != nil then
        @signed = hash[:signed]
      end
      @size = nil
      @vector_length = 1
    end

    def to_hash
      return { :signed => @signed }
    end

    def copy(options={})
      hash = self.to_hash
      options.each { |k,v|
        hash[k] = v
      }
      return Sizet::new(hash)
    end

    def decl
      return "integer(kind=#{BOAST::get_default_int_size})" if BOAST::get_lang == FORTRAN
      if not @signed then
        return "size_t" if [C, CL, CUDA].include?( BOAST::get_lang )
      else
        return "ptrdiff_t" if [C, CL, CUDA].include?( BOAST::get_lang )
      end
    end
  end
 
  class Real
    extend BOAST::VarFunctor

    attr_reader :size
    attr_reader :signed
    attr_reader :vector_length
    attr_reader :total_size
    attr_reader :getters
    attr_reader :setters

    def ==(t)
      return true if t.class == self.class and t.size == self.size and t.vector_length == self.vector_length
      return false
    end

    def initialize(hash={})
      if hash[:size] then
        @size = hash[:size]
      else
        @size = BOAST::get_default_real_size
      end
#      @getters = {}
#      @setters = {}
      if hash[:vector_length] and hash[:vector_length] > 1 then
        @vector_length = hash[:vector_length]
#        @vector_length.times{ |indx|
#          @getters["s#{indx}"] = indx
#          @setters["s#{indx}="] = indx
#        }
      else
        @vector_length = 1
      end
      @total_size = @vector_length*@size
      @signed = true
    end

    def to_hash
      return { :size => @size, :vector_length => @vector_length }
    end

    def copy(options={})
      hash = to_hash
      options.each { |k,v|
        hash[k] = v
      }
      return Real::new(hash)
    end

    def decl
      return "real(kind=#{@size})" if BOAST::get_lang == FORTRAN
      if [C, CL, CUDA].include?( BOAST::get_lang ) and @vector_length == 1 then
        return "float" if @size == 4
        return "double" if @size == 8
      elsif BOAST::get_lang == C and @vector_length > 1 then
        if BOAST::get_architecture == BOAST::X86 then
          return "__m#{@total_size*8}" if @size == 4
          return "__m#{@total_size*8}d" if @size == 8
        elsif BOAST::get_architecture == BOAST::ARM then
          raise "Unsupported data type in NEON: double!" if @size == 8
          raise "Unsupported vector length in NEON: #{@total_size} (#{@size} x 8 x #{@vector_length})!" if @total_size * 8 != 64 or @total_size * 8 != 128
          return "float#{@size*8}x#{@vector_length}_t"
        else
          raise "Unsupported architecture!"
        end
      elsif [CL, CUDA].include?( BOAST::get_lang ) and @vector_length > 1 then
        return "float#{@vector_length}" if @size == 4
        return "double#{@vector_length}" if @size == 8
      end
    end
  end

  class Int
    extend BOAST::VarFunctor

    attr_reader :size
    attr_reader :signed
    attr_reader :vector_length
    attr_reader :total_size

    def ==(t)
      return true if t.class == self.class and t.signed == self.signed and t.size == self.size and t.vector_length == self.vector_length
      return false
    end

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
#      @getters = {}
#      @setters = {}
      if hash[:vector_length] and hash[:vector_length] > 1 then
        @vector_length = hash[:vector_length]
#        @vector_length.times{ |indx|
#          @getters["s#{indx}"] = indx
#          @setters["s#{indx}="] = indx
#        }
      else
        @vector_length = 1
      end
      @total_size = @vector_length*@size
    end

    def to_hash
      return { :size => @size, :vector_length => @vector_length, :signed => @signed }
    end

    def copy(options={})
      hash = self.to_hash
      options.each { |k,v|
        hash[k] = v
      }
      return Int::new(hash)
    end

    def signed?
      return @signed
    end

    def decl
      return "integer(kind=#{@size})" if BOAST::get_lang == FORTRAN
      if BOAST::get_lang == C then
        if @vector_length == 1 then
          s = ""
          s += "u" if not @signed
          return s+"int#{8*@size}_t"
        elsif @vector_length > 1 then
          if BOAST::get_architecture == BOAST::X86 then
            return "__m#{@total_size*8}#{@total_size*8>64 ? "i" : ""}"
          elsif BOAST::get_architecture == BOAST::ARM then
            raise "Unsupported vector length in NEON: #{@total_size*8} (#{@size} x 8 x #{@vector_length})!" if @total_size * 8 != 64 and @total_size * 8 != 128
            return "#{ @signed ? "" : "u"}int#{@size*8}x#{@vector_length}_t"
          else
            raise "Unsupported architecture!"
          end
        end
      elsif BOAST::get_lang == CL then
        #char="cl_"
        char=""
        char += "u" if not @signed
        case @size
        when 1
          char += "char"
        when 2
          char += "short"
        when 4
          char += "int"
        when 8
          char += "long"
        else
          raise "Unsupported integer size!"
        end
        if @vector_length > 1 then
          char += "#{@vector_length}"
        end
        return char
      elsif BOAST::get_lang == CUDA then
        if @vector_length > 1 then
          char=""
          char += "u" if not @signed
          case @size
          when 1
            char += "char"
          when 2
            char += "short"
          when 4
            char += "int"
          when 8
            char += "longlong"
          else
            raise "Unsupported integer size!"
          end
          return char + "#{@vector_length}"
        else
          char = ""
          char += "unsigned " if not @signed
          return char += "char" if @size==1
          return char += "short" if @size==2
          return char += "int" if @size==4
          return char += "long long" if @size==8
        end
      end
    end
  end

  class CStruct
    extend BOAST::VarFunctor

    attr_reader :name, :members, :members_array

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

    def decl_c
      return "struct #{@name}" if [C, CL, CUDA].include?( BOAST::get_lang )
    end

    def decl_fortran
      return "TYPE(#{@name})" if BOAST::get_lang == FORTRAN
    end

    def decl
      return self.decl_c if [C, CL, CUDA].include?( BOAST::get_lang )
      return self.decl_fortran if BOAST::get_lang == FORTRAN
    end

    def finalize
       s = ""
       s += ";" if [C, CL, CUDA].include?( BOAST::get_lang )
       s+="\n"
       return s
    end

    def define
      return define_c if [C, CL, CUDA].include?( BOAST::get_lang )
      return define_fortran if BOAST::get_lang == FORTRAN
    end

    def define_c
      s = BOAST::indent
      s += self.decl_c + " {"
      BOAST::get_output.puts s
      @members_array.each { |value|
         value.decl
      }
      s = BOAST::indent
      s += "}"
      s += self.finalize
      BOAST::get_output.print s
      return self
    end
    
    def define_fortran
      s = BOAST::indent
      s += "TYPE :: #{@name}\n"
      BOAST::get_output.puts s
      @members_array.each { |value|
         value.decl
      }
      s = BOAST::indent
      s += "END TYPE #{@name}"
      s += self.finalize
      BOAST::get_output.print s
      return self
    end

  end

  class CustomType
    extend BOAST::VarFunctor

    attr_reader :size, :name, :vector_length
    def initialize(hash={})
      @name = hash[:type_name]
      @size = hash[:size]
      @size = 0 if @size.nil?
      @vector_length = hash[:vector_length]
      @vector_length = 1 if @vector_length.nil?
      @total_size = @vector_length*@size
    end
    def decl
      return "#{@name}" if [C, CL, CUDA].include?( BOAST::get_lang )
    end
  end

end
