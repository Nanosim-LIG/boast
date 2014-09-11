module BOAST

  class Operator
    extend PrivateStateAccessor

    def Operator.inspect
      return "#{name}"
    end

    def Operator.get_vector_name(type)
      case get_architecture
      when X86
        case type
        when Int
          size = "#{type.size*8}"
          name = ""
          if type.total_size*8 > 64
            name += "e"
          end
          if type.vector_length > 1 then
            name += "p"
          else
            name = "s"
          end
          if type.signed then
            name += "i"
          else
            name += "u"
          end
          return name += size
        when Real
          case type.size
          when 4
            return "ps" if type.vector_length > 1
            return "ss"
          when 8
            return "pd" if type.vector_length > 1
            return "sd"
          end
        else
          raise "Undefined vector type!"
        end
      when ARM
        case type
        when Int
          name = "#{ type.signed ? "s" : "u" }"
          name += "#{ type.size * 8}"
          return name
        when Real
          return "f#{type.size*8}"
        else
          raise "Undefined vector type!"
        end
      else
        raise "Unsupported architecture!"
      end
    end

    def Operator.convert(arg, type)
      case get_architecture
      when X86
        s1 = arg.type.total_size*8
        s2 = type.total_size*8
        n1 = get_vector_name(arg.type)
        n2 = get_vector_name(type)
        if s1 <= 128 and s2 <= 128 then
          return "_mm_cvt#{n1}_#{n2}( #{arg} )"
        elsif [s1, s2].max <= 256 then
          return "_mm256_cvt#{n1}_#{n2}( #{arg} )"
        elsif [s1, s2].max <= 512 then
          return "_mm512_cvt#{n1}_#{n2}( #{arg} )"
        end
      when ARM
        if type.class != arg.type.class then
          if type.size == arg.type.size then
            s = type.total_size*8
            n1 = get_vector_name(arg.type)
            n2 = get_vector_name(type)
            return "vcvt#{ s == 128 ? "q" : "" }_#{n2}_#{n1}( #{arg} )"
          elsif type.class == Real then
            intr = convert(arg, arg.type.copy(:size=>type.size))
            return convert(arg.copy(intr, :size => type.size ), type)
          else
            n1 = get_vector_name(arg.type)
            s = type.total_size*8
            t2 = type.copy(:size => arg.type.size)
            n2 = get_vector_name( t2 )
            intr = "vcvt#{ s == 128 ? "q" : "" }_#{n2}_#{n1}( #{arg} )"
            return convert(Variable::from_type(intr, t2), type)
          end
        elsif type.class != Real then
          n = get_vector_name(arg.type)
          if type.size == arg.type.size then
            if type.signed == arg.type.signed then
              return "#{arg}"
            else
              n2 = get_vector_name(type)
              return "vreinterpret_#{n2}_#{n}( #{arg} )"
            end
          elsif type.size < arg.type.size then
            intr = "vmovn_#{n}( #{arg} )"
            s = arg.type.size/2
          else
            intr = "vmovl_#{n}( #{arg} )"
            s = arg.type.size*2
          end
          return convert(arg.copy(intr, :size => s), type)
        end
      else
        raise "Unsupported architecture!"
      end
    end

  end

  class BasicBinaryOperator < Operator

    def BasicBinaryOperator.to_s(arg1, arg2, return_type)
      #puts "#{arg1.class} * #{arg2.class} : #{arg1} * #{arg2}"
      if lang == C and (arg1.class == Variable and arg2.class == Variable) and (arg1.type.vector_length > 1 or arg2.type.vector_length > 1) then
        raise "Vectors have different length: #{arg1} #{arg1.type.vector_length}, #{arg2} #{arg2.type.vector_length}" if arg1.type.vector_length != arg2.type.vector_length
        #puts "#{arg1.type.signed} #{arg2.type.signed} #{return_type.type.signed}"
	return_name = get_vector_name(return_type.type)
        size = return_type.type.total_size * 8
        case get_architecture
        when X86
          if arg1.type != return_type.type
            a1 = convert(arg1, return_type.type)
          else
            a1 = "#{arg1}"
          end
          if arg2.type != return_type.type
            a2 = convert(arg2, return_type.type)
          else
            a2 = "#{arg2}"
          end
          intr_name = "_mm"
          if size > 128 then
            intr_name += "#{size}"
          end
          intr_name += "_#{intr_name_X86}_#{return_name}"
          return "#{intr_name}( #{a1}, #{a2} )"
        when ARM
          if arg1.type.class != return_type.type.class then
            a1 = convert(arg1, return_type.type)
          else
            a1 = "#{arg1}"
          end
          if arg2.type.class != return_type.type.class then
            a2 = convert(arg2, return_type.type)
          else
            a2 = "#{arg2}"
          end
          intr_name = "#{intr_name_ARM}"
          intr_name += "q" if size == 128
          intr_name += "_" + return_name + "( #{a1}, #{a2} )"
          return intr_name
        else
          raise "Unsupported architecture!"
        end
      else
        return basic_usage( arg1, arg2 )
      end
    end

  end

  class Set < Operator

    def Set.to_s(arg1, arg2, return_type)
      if lang == C then
        if arg1.class == Variable and arg1.type.vector_length > 1 then
          if arg1.type == arg2.type then
            return basic_usage(arg1, arg2)
          elsif arg1.type.vector_length == arg2.type.vector_length then
            return "(#{arg1} = #{convert(arg2, arg1.type)})"
          elsif arg2.type.vector_length == 1 then
            size = arg1.type.total_size*8
            case get_architecture
            when ARM
              intr_name = "vmov"
              intr_name += "q" if size == 128
              intr_name += "_n_#{get_vector_name(arg1.type)}"
            when X86
              return "(#{arg1} = _m_from_int64( #{a2} ))" if arg1.type.class == Int and arg1.type.size == 8 and size == 64
              intr_name = "_mm"
              if size > 128 then
                intr_name += "#{size}"
              end
              intr_name += "_set1_#{get_vector_name(arg1.type).gsub("u","")}"
              intr_name += "x" if arg1.type.class == Int and arg1.type.size == 8
            else
              raise "Unsupported architecture!"
            end
            return "(#{arg1} = #{intr_name}( #{arg2} ))"
          else
            raise "Unknown convertion between vector of different length!"
          end
        else
          return basic_usage(arg1, arg2)
        end
      else
        return basic_usage(arg1, arg2)
      end
    end

    def Set.basic_usage(arg1, arg2)
      return "(#{arg1} = #{arg2})"
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

  class Affectation < Operator

    def Affectation.to_s(arg1, arg2, return_type)
      if lang == C then
        if arg1.class == Variable and arg1.type.vector_length > 1 then
          #puts "#{arg1.type.vector_length} #{arg2.type.vector_length}"
          if arg1.type == arg2.type then
            return basic_usage(arg1, arg2)
          elsif arg1.type.vector_length == arg2.type.vector_length then
            return "#{arg1} = #{convert(arg2, arg1.type)}"
          elsif arg2.type.vector_length == 1 then
            size = arg1.type.total_size*8
            a2 = "#{arg2}"
            if a2[0] != "*" then
              a2 = "&" + a2
            else
              a2 = a2[1..-1]
            end
            case get_architecture
            when ARM
              intr_name = "vldl"
              intr_name += "q" if size == 128
              intr_name += "_#{get_vector_name(arg1.type)}"
            when X86
              if arg1.type.class == Int and size == 64 then
                return "#{arg1} = _m_from_int64( *((int64_t * ) #{a2} ) )"
              end
              intr_name = "_mm"
              if size > 128 then
                intr_name += "#{size}"
              end
              intr_name += "_load_"
              if arg1.type.class == Int then
                intr_name += "si#{size}"
              else
                intr_name += "#{get_vector_name(arg1.type)}"
              end
            else
              raise "Unsupported architecture!"
            end
            return "#{arg1} = #{intr_name}( (#{arg1.type.decl} * ) #{a2} )"
          else
            raise "Unknown convertion between vectors of different length!"
          end
        elsif arg2.class == Variable and arg2.type.vector_length > 1 then
          size = arg2.type.total_size*8
          a1 = "#{arg1}"
          if a1[0] != "*" then
            a1 = "&" + a1
          else
            a1 = a1[1..-1]
          end
          case get_architecture
          when ARM
            intr_name = "vstl"
            intr_name += "q" if size == 128
            intr_name += "_#{get_vector_name(arg2.type)}"
          when X86
            if arg2.type.class == Int and size == 64 then
              return " *((int64_t * ) #{a1}) = _m_to_int64( #{arg2} )"
            end
            intr_name = "_mm"
            if size > 128 then
              intr_name += "#{size}"
            end
            intr_name += "_store_"
            if arg2.type.class == Int then
              intr_name += "si#{size}"
            else
              intr_name += "#{get_vector_name(arg2.type)}"
            end
          else
            raise "Unsupported architecture!"
          end
          return "#{intr_name}((#{arg2.type.decl} * ) #{a1}, #{arg2} )"
        else
          return basic_usage(arg1, arg2)
        end
      else
        return basic_usage(arg1, arg2)
      end
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

      def intr_name_X86
        return "mul"
      end

      def intr_name_ARM
        return "vmul"
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

      def intr_name_X86
        return "add"
      end
  
      def intr_name_ARM
        return "vadd"
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

      def intr_name_X86
        return "sub"
      end
  
      def intr_name_ARM
        return "vsub"
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

      def intr_name_X86
        return "div"
      end
  
      def intr_name_ARM
        raise "Neon doesn't support division!"
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
