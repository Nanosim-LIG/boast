module BOAST

  class Operator
    extend PrivateStateAccessor
    extend Intrinsics

    def Operator.inspect
      return "#{name}"
    end

    def Operator.convert(arg, type)
      return "convert_#{type.decl}( #{arg} )" if lang == CL

      path = get_conversion_path(type, arg.type)
      raise "Unavailable conversion from #{get_vector_name(arg.type)} to #{get_vector_name(type)}!" if not path
      s = "#{arg}"
      if path.length > 1 then
        path.each_cons(2) { |slice|
          instruction = intrinsics_by_vector_name(:CVT, slice[1], slice[0])
          s = "#{instruction}( #{s} )"
        }
      end
      return s
    end

  end

  class BasicBinaryOperator < Operator

    def BasicBinaryOperator.to_s(arg1, arg2, return_type)
      if lang == C and (arg1.class == Variable and arg2.class == Variable) and (arg1.type.vector_length > 1 or arg2.type.vector_length > 1) then
        instruction = intrinsics(intr_symbol, return_type.type)
        raise "Unavailable operator #{symbol} for #{get_vector_name(return_type.type)}!" unless instruction
        a1 = convert(arg1, return_type.type)
        a2 = convert(arg2, return_type.type)
        return "#{instruction}( #{a1}, #{a2} )"
      else
        return basic_usage( arg1, arg2 )
      end
    end

  end

  class Different < Operator

    def Different.to_s(arg1, arg2, return_type)
      return basic_usage(arg1, arg2)
    end

    def Different.basic_usage(arg1, arg2)
      return "#{arg1} /= #{arg2}" if lang == FORTRAN
      return "#{arg1} != #{arg2}"
    end

  end

  class Set < Operator

    def Set.to_s(arg1, arg2, return_type)
      s = ""
      s += "#{arg1} = " if arg1
      return_type = arg1.to_var unless return_type
      if lang == C or lang == CL and return_type.type.vector_length > 1 then
        if arg2.kind_of?( Array ) then
          raise "Invalid array length!" unless arg2.length == return_type.type.vector_length
          return s += "(#{return_type.type.decl})( #{arg2.join(", ")} )" if lang == CL

          instruction = intrinsics(:SET, return_type.type)
          return s += "#{instruction}( #{arg2.join(", ")} )" if instruction

          instruction = intrinsics(:SET_LANE, return_type.type)
          raise "Unavailable operator set for #{get_vector_name(return_type.type)}!" unless instruction
          ss = Set.to_s(0, return_type)
          arg2.each_with_index { |v,i|
            ss = "#{instruction}(#{v}, #{ss}, #{i})"
          }
          return s += ss
        elsif arg2.class != Variable or arg2.type.vector_length == 1 then
          return s += "(#{return_type.type.decl})( #{arg2} )" if lang == CL

          instruction = intrinsics(:SET1, return_type.type)
          raise "Unavailable operator set1 for #{get_vector_name(return_type.type)}!" unless instruction
          return s += "#{instruction}( #{arg2} )"
        elsif return_type.type != arg2.type
          return s+= "#{convert(arg2, return_type.type)}"
        end
      end
      return s += "#{arg2}"
    end

  end

  class Load < Operator

    def Load.to_s(arg2, return_type)
      if lang == C or lang == CL then
        if arg2.kind_of?(Array) then
          return Set.to_s(nil, arg2, return_type)
        elsif arg2.class == Variable or arg2.respond_to?(:to_var) then
          if arg2.to_var.type == return_type.type
            return "#{arg2}"
          elsif arg2.type.vector_length == 1 then
            a2 = "#{arg2}"
            if a2[0] != "*" then
              a2 = "&" + a2
            else
              a2 = a2[1..-1]
            end
            return "vload#{return_type.type.vector_length}(0, #{a2})" if lang == CL
            return "_m_from_int64( *((int64_t * ) #{a2} ) )" if get_architecture == X86 and return_type.type.total_size*8 == 64
            if arg2.align == return_type.type.total_size then
              instruction = intrinsics(:LOADA, return_type.type)
            else
              instruction = intrinsics(:LOAD, return_type.type)
            end
            raise "Unavailable operator load for #{get_vector_name(return_type.type)}!" unless instruction
            return "#{instruction}( #{a2} )"
          else
            return "#{convert(arg2, return_type.type)}"
          end
        end
      end
      return "#{arg2}"
    end

  end

  class Store < Operator

    def Store.to_s(arg1, arg2, return_type)
      if lang == C or lang == CL then
        a1 = "#{arg1}"
        if a1[0] != "*" then
          a1 = "&" + a1
        else
          a1 = a1[1..-1]
        end

        return "vstore#{arg2.type.vector_length}(#{arg2}, 0, #{a1})" if lang == CL
        return "*((int64_t * ) #{a1}) = _m_to_int64( #{arg2} )" if get_architecture == X86 and arg2.type.total_size*8 == 64

        if arg1.align == arg2.type.total_size then
          instruction = intrinsics(:STOREA, arg2.type)
        else
          instruction = intrinsics(:STORE, arg2.type)
        end
        raise "Unavailable operator store for #{get_vector_name(arg2.type)}!" unless instruction
        p_type = arg2.type.copy(:vector_length => 1)
        p_type = arg2.type if get_architecture == X86 and arg2.type.kind_of?(Int)
        return "#{instruction}( (#{p_type.decl} * ) #{a1}, #{arg2} )"
      end
      return Affectation.basic_usage(arg1, arg2)
    end

  end

  class Affectation < Operator

    def Affectation.to_s(arg1, arg2, return_type)
      if arg1.class == Variable and arg1.type.vector_length > 1 then
        return "#{arg1} = #{Load.to_s(arg2, arg1)}"
      elsif arg2.class == Variable and arg2.type.vector_length > 1 then
        return Store.to_s(arg1, arg2, return_type)
      end
      return basic_usage(arg1, arg2)
    end

    def Affectation.basic_usage(arg1, arg2)
      return "#{arg1} = #{arg2}"
    end

  end

  class Multiplication < BasicBinaryOperator

    class << self

      def symbol
        return "*"
      end

      def intr_symbol
        return :MUL
      end

      def basic_usage(arg1, arg2)
        return "(#{arg1}) * (#{arg2})" 
      end
  
    end

  end

  class Addition < BasicBinaryOperator

    class << self

      def symbol
        return "+"
      end

      def intr_symbol
        return :ADD
      end
  
      def basic_usage(arg1, arg2)
        return "#{arg1} + #{arg2}" 
      end
  
    end

  end

  class Substraction < BasicBinaryOperator

    class << self

      def symbol
        return "-"
      end

      def intr_symbol
        return :SUB
      end
  
      def basic_usage(arg1, arg2)
        return "#{arg1} - (#{arg2})" 
      end
  
    end

  end

  class Division < BasicBinaryOperator

    class << self

      def symbol
        return "/"
      end

      def intr_symbol
        return :DIV
      end
  
      def basic_usage(arg1, arg2)
        return "(#{arg1}) / (#{arg2})" 
      end
  
    end

  end

  class Minus < Operator

    def Minus.to_s(arg1, arg2, return_type)
      return " -(#{arg2})"
    end

  end

  class Not < Operator

    def Not.to_s(arg1, arg2, return_type)
      return " ! #{arg2}"
    end

  end

  class MaskLoad < Operator
    extend Functor
    include Intrinsics
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :source
    attr_reader :mask
    attr_reader :return_type

    def initialize(source, mask, return_type)
      @source = source
      @mask = mask
      @return_type = return_type
    end

    def get_mask
      raise "Mask size is wrong: #{@mask.length} for #{@return_type.type.vector_length}!" if @mask.length != @return_type.type.vector_length
      return Load.to_s(@mask.collect { |m| ( m and m != 0 )  ? -1 : 0 }, Int("mask", :size => @return_type.type.size, :vector_length => @return_type.type.vector_length ) )
    end

    private :get_mask

    def to_var
      raise "Cannot load unknown type!" unless @return_type
      raise "Unsupported language!" unless lang == C
      raise "MaskLoad not supported!"unless supported(:MASKLOAD, @return_type.type)
      s = ""
      src = "#{@source}"
      if src[0] != "*" then
        src = "&" + src
      else
        src = src[1..-1]
      end
      s += "#{intrinsics(:MASKLOAD, @return_type.type)}(#{src}, #{get_mask})"
      return @return_type.copy( s, :const => nil, :constant => nil, :direction => nil, :dir => nil, :align => nil)
    end

    def to_s
      return to_var.to_s
    end

    def pr
      s=""
      s += indent
      s += to_s
      s += ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end

  class MaskStore < Operator
    extend Functor
    include Intrinsics
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :dest
    attr_reader :source
    attr_reader :mask

    def initialize(dest, source, mask, store_type = nil)
      @dest = dest
      @source = source
      @mask = mask
      @store_type = store_type
      @store_type = source unless @store_type
    end

    def get_mask
      raise "Mask size is wrong: #{@mask.length} for #{@store_type.type.vector_length}!" if @mask.length != @store_type.type.vector_length
      return Load.to_s(@mask.collect { |m| ( m and m != 0 )  ? -1 : 0 }, Int("mask", :size => @store_type.type.size, :vector_length => @store_type.type.vector_length ) )
    end

    private :get_mask

    def to_s
      raise "Cannot store unknown type!" unless @store_type
      raise "Unsupported language!" unless lang == C
      raise "MaskStore not supported!"unless supported(:MASKSTORE, @store_type.type)
      s = ""
      dst = "#{@dest}"
      if dst[0] != "*" then
        dst = "&" + dst
      else
        dst = dst[1..-1]
      end
      p_type = @store_type.type.copy(:vector_length => 1)
      return s += "#{intrinsics(:MASKSTORE, @store_type.type)}((#{p_type.decl} * )#{dst}, #{Operator.convert(@source, @store_type.type)}, #{get_mask})"
    end

    def pr
      s=""
      s += indent
      s += to_s
      s += ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end

  class FMA < Operator
    extend Functor
    include Intrinsics
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :operand1
    attr_reader :operand2
    attr_reader :operand3
    attr_reader :return_type

    def initialize(a,b,c)
      @operand1 = a
      @operand2 = b
      @operand3 = c
      @return_type = nil
    end

    def get_return_type
      @return_type = @operand3.to_var
    end

    private :get_return_type

    def convert_operand(op)
      return  "#{Operator.convert(op, @return_type.type)}"
    end

    private :convert_operand

    def to_var
      get_return_type
      return (@operand1 * @operand2 + @operand3).to_var unless lang != FORTRAN and @return_type and ( supported(:FMADD, @return_type.type) or ( [CL, CUDA].include?(lang) ) )
      op1 = convert_operand(@operand1)
      op2 = convert_operand(@operand2)
      op3 = convert_operand(@operand3)
      if [CL, CUDA].include?(lang)
        ret_name = "fma(#{op1},#{op2},#{op3})"
      else
        case architecture
        when X86
          ret_name = "#{intrinsics(:FMADD,@return_type.type)}(#{op1},#{op2},#{op3})"
        when ARM
          ret_name = "#{intrinsics(:FMADD,@return_type.type)}(#{op2},#{op3},#{op1})"
        else
          return (@operand1 * @operand2 + @operand3).to_var
        end
      end
      return @return_type.copy( ret_name, :const => nil, :constant => nil, :direction => nil, :dir => nil, :align => nil)
    end

    def to_s
      return to_var.to_s
    end

    def pr
      s=""
      s += indent
      s += to_s
      s += ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end

  class Ternary
    extend Functor
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :operand1
    attr_reader :operand2
    attr_reader :operand3
    
    def initialize(x,y,z)
      @operand1 = x
      @operand2 = y
      @operand3 = z
    end

    def to_s
      raise "Ternary operator unsupported in FORTRAN!" if lang == FORTRAN
      return to_s_c if [C, CL, CUDA].include?( lang )
    end

    def to_s_c
      s = ""
      s += "(#{@operand1} ? #{@operand2} : #{@operand3})"
    end

    def pr
      s=""
      s += indent
      s += to_s
      s += ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end

end
