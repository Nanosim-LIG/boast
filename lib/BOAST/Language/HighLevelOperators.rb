module BOAST

  class HighLevelOperator < Operator
    include Intrinsics
    include Arithmetic
    include Inspectable
  end

  class Sqrt < HighLevelOperator
    extend Functor

    attr_reader :operand
    attr_reader :return_type

    def initialize(a)
      @operand = a
      @return_type = a.to_var
      unless @return_type.type.kind_of?(Real) then
        @return_type = Variable::new(:sqrt_type, Real, :vector_length => @return_type.type.vector_length)
      end
    end

    def convert_operand(op)
      return  "#{Operator.convert(op, @return_type.type)}"
    end

    private :convert_operand

    def type
      return @return_type.type
    end

    def to_var
      sqrt_instruction = nil
      rsqrt_instruction = nil
      begin
        sqrt_instruction = intrinsics(:SQRT,@return_type.type)
      rescue
      end
      unless sqrt_instruction then
        begin
          rsqrt_instruction = intrinsics(:RSQRT,@return_type.type)
        rescue
        end
      end

      if [FORTRAN, CL].include?(lang) then
        return @return_type.copy( "sqrt( #{@operand} )", DISCARD_OPTIONS )
      elsif lang == CUDA or ( sqrt_instruction.nil? and rsqrt_instruction.nil? ) then
        raise IntrinsicsError, "Vector square root unsupported on ARM architecture!" if architecture == ARM and @return_type.type.vector_length > 1
        if @return_type.type.size <= 4 then
          return @return_type.copy( "sqrtf( #{@operand} )", DISCARD_OPTIONS )
        else
          return @return_type.copy( "sqrt( #{@operand} )", DISCARD_OPTIONS )
        end
      end
      op = convert_operand(@operand.to_var)
      if sqrt_instruction then
        return @return_type.copy( "#{sqrt_instruction}( #{op} )", DISCARD_OPTIONS )
      else
        return (op * @return_type.copy("#{rsqrt_instruction}( #{op} )", DISCARD_OPTIONS)).to_var
      end
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

  class TrigonometricOperator < HighLevelOperator
    attr_reader :operand
    attr_reader :return_type

    def initialize(a)
      @operand = a
      @return_type = a.to_var
      unless @return_type.type.kind_of?(Real) then
        @return_type = Variable::new(:trig_type, Real, :vector_length => @return_type.type.vector_length)
      end
    end

    def convert_operand(op)
      return  "#{Operator.convert(op, @return_type.type)}"
    end

    private :convert_operand

    def type
      return @return_type.type
    end

    def to_var
      instruction = nil
      begin
        instruction = intrinsics(get_intrinsic_symbol,@return_type.type)
      rescue
      end

      if [FORTRAN, CL].include?(lang) then
        return @return_type.copy( "#{get_name[lang]}( #{@operand} )", DISCARD_OPTIONS )
      elsif lang == CUDA or instruction.nil? then
        raise IntrinsicsError, "Vector #{get_name[lang]} root unsupported on ARM architecture!" if architecture == ARM and @return_type.type.vector_length > 1
        if @return_type.type.size <= 4 then
          return @return_type.copy( "#{get_name[lang]}f( #{@operand} )", DISCARD_OPTIONS )
        else
          return @return_type.copy( "#{get_name[lang]}( #{@operand} )", DISCARD_OPTIONS )
        end
      end
      op = convert_operand(@operand.to_var)
      return @return_type.copy( "#{instruction}( #{op} )", DISCARD_OPTIONS )
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

  def self.generic_trigonometric_operator_generator( name )
    eval <<EOF
  class #{name.capitalize} < TrigonometricOperator
    extend Functor

    def get_intrinsic_symbol
      return :#{name.upcase}
    end

    def get_name
      return { C => "#{name}", CUDA => "#{name}", CL => "#{name}", FORTRAN => "#{name}" }
    end

  end

EOF
  end

  generic_trigonometric_operator_generator( "sin" )
  generic_trigonometric_operator_generator( "cos" )
  generic_trigonometric_operator_generator( "tan" )
  generic_trigonometric_operator_generator( "sinh" )
  generic_trigonometric_operator_generator( "cosh" )
  generic_trigonometric_operator_generator( "tanh" )
  generic_trigonometric_operator_generator( "asin" )
  generic_trigonometric_operator_generator( "acos" )
  generic_trigonometric_operator_generator( "atan" )
  generic_trigonometric_operator_generator( "asinh" )
  generic_trigonometric_operator_generator( "acosh" )
  generic_trigonometric_operator_generator( "atanh" )

  generic_trigonometric_operator_generator( "exp" )
  generic_trigonometric_operator_generator( "log" )
  generic_trigonometric_operator_generator( "log10" )
end
