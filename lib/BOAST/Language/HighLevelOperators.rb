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

end
