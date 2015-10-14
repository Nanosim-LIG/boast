module BOAST

  class DataType
    include Intrinsics
    include PrivateStateAccessor

    def self.inherited(child)
      child.extend( VarFunctor)
    end

  end

  class Sizet < DataType

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
      hash = to_hash
      options.each { |k,v|
        hash[k] = v
      }
      return Sizet::new(hash)
    end

    def decl
      return "integer(kind=#{get_default_int_size})" if lang == FORTRAN
      if not @signed then
        return "size_t" if [C, CL, CUDA].include?( lang )
      else
        return "ptrdiff_t" if [C, CL, CUDA].include?( lang )
      end
    end

    def decl_ffi
      return :size_t
    end

    def signed?
      return !!signed
    end

    def suffix
      s = ""
      return s
    end

  end
 
  class Real < DataType

    attr_reader :size
    attr_reader :signed
    attr_reader :vector_length
    attr_reader :total_size
    attr_reader :getters
    attr_reader :setters

    def ==(t)
      return true if t.class == self.class and t.size == size and t.vector_length == vector_length
      return false
    end

    def initialize(hash={})
      if hash[:size] then
        @size = hash[:size]
      else
        @size = get_default_real_size
      end
      if hash[:vector_length] and hash[:vector_length] > 1 then
        @vector_length = hash[:vector_length]
      else
        @vector_length = 1
      end
      @total_size = @size*@vector_length
      @signed = true
    end

    def signed?
      return !!signed
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
      return "real(kind=#{@size})" if lang == FORTRAN
      if [C, CL, CUDA].include?( lang ) and @vector_length == 1 then
        return "float" if @size == 4
        return "double" if @size == 8
      elsif lang == C and @vector_length > 1 then
        return get_vector_decl(self)
      elsif [CL, CUDA].include?( lang ) and @vector_length > 1 then
        return "float#{@vector_length}" if @size == 4
        return "double#{@vector_length}" if @size == 8
      end
    end

    def decl_ffi
      return :float if @size == 4
      return :double if @size == 8
    end

    def suffix
      s = ""
      if [C, CL, CUDA].include?( lang ) then
        s += "f" if @size == 4
      elsif lang == FORTRAN then
        s += "_wp" if @size == 8
      end
      return s
    end

  end

  class Int < DataType

    attr_reader :size
    attr_reader :signed
    attr_reader :vector_length
    attr_reader :total_size

    def ==(t)
      return true if t.class == self.class and t.signed == signed and t.size == size and t.vector_length == vector_length
      return false
    end

    def initialize(hash={})
      if hash[:size] then
        @size = hash[:size]
      else
        @size = get_default_int_size
      end
      if hash[:signed] != nil then
        @signed = hash[:signed]
      else
        @signed = get_default_int_signed
      end
      if hash[:vector_length] and hash[:vector_length] > 1 then
        @vector_length = hash[:vector_length]
        raise "Vectors need to have their element size specified!" if not @size
      else
        @vector_length = 1
      end
      @total_size = @size*@vector_length
    end

    def to_hash
      return { :size => @size, :vector_length => @vector_length, :signed => @signed }
    end

    def copy(options={})
      hash = to_hash
      options.each { |k,v|
        hash[k] = v
      }
      return Int::new(hash)
    end

    def signed?
      return !!@signed
    end

    def decl
      if lang == FORTRAN then
        return "integer(kind=#{@size})" if @size
        return "integer"
      end
      if lang == C then
        if @vector_length == 1 then
          s = ""
          s += "u" if not @signed
          return s+"int#{8*@size}_t" if @size
          return s+"int"
        elsif @vector_length > 1 then
          return get_vector_decl(self)
        end
      else
        s =""
        s += "u" if not @signed
        s += "nsigned " if not @signed and lang == CUDA and @vector_length == 1
        case @size
        when 1
          s += "char"
        when 2
          s += "short"
        when 4
          s += "int"
        when 8
          if lang == CUDA
            case @vector_length
            when 1
              s += "long long"
            else
              s += "longlong"
            end
          else
            s += "long"
          end
        when nil
          s += "int"
        else
          raise "Unsupported integer size!"
        end
        if @vector_length > 1 then
          s += "#{@vector_length}"
        end
        return s
      end
    end

    def decl_ffi
      t = ""
      t += "u" if not @signed
      t += "int"
      t += "#{@size*8}" if @size
      return t.to_sym
    end

    def suffix
      s = ""
      return s
    end

  end

  class CStruct < DataType

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
      return "struct #{@name}" if [C, CL, CUDA].include?( lang )
    end

    def decl_fortran
      return "TYPE(#{@name})" if lang == FORTRAN
    end

    def decl
      return decl_c if [C, CL, CUDA].include?( lang )
      return decl_fortran if lang == FORTRAN
    end

    def finalize
       s = ""
       s += ";" if [C, CL, CUDA].include?( lang )
       s+="\n"
       return s
    end

    def define
      return define_c if [C, CL, CUDA].include?( lang )
      return define_fortran if lang == FORTRAN
    end

    def define_c
      s = indent
      s += decl_c + " {"
      output.puts s
      increment_indent_level
      @members_array.each { |value|
         value.decl
      }
      decrement_indent_level
      s = indent
      s += "}"
      s += finalize
      output.print s
      return self
    end
    
    def define_fortran
      s = indent
      s += "TYPE :: #{@name}\n"
      output.puts s
      increment_indent_level
      @members_array.each { |value|
         value.decl
      }
      decrement_indent_level
      s = indent
      s += "END TYPE #{@name}"
      s += finalize
      output.print s
      return self
    end

  end

  class CustomType < DataType

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
      return "#{@name}" if [C, CL, CUDA].include?( lang )
    end

  end

end
