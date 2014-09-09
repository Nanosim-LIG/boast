module BOAST

  def Return(value)
    return Expression("return",nil, value)
  end

  extend self

  class Expression
    include BOAST::Arithmetic
    include BOAST::Inspectable
    extend BOAST::Functor

    attr_reader :operator
    attr_reader :operand1
    attr_reader :operand2
    def initialize(operator, operand1, operand2)
      @operator = operator
      @operand1 = operand1
      @operand2 = operand2
    end

    def Expression.to_s_base(op1, op2, oper, return_type = nil)
      return oper.to_s(op1, op2, return_type) if not oper.kind_of?(String)
      s = ""
      if op1 then
        s += "(" if (oper == "*" or oper == "/") 
        s += op1.to_s
        s += ")" if (oper == "*" or oper == "/") 
      end        
      s += " " unless oper == "++" or oper == "."
      s += oper unless ( oper == "&" and BOAST::lang == BOAST::FORTRAN )
      s += " " unless oper == "." or oper == "&" or ( oper == "*" and op1.nil? )
      if op2 then
        s += "(" if (oper == "*" or oper == "/" or oper == "-") 
        s += op2.to_s
        s += ")" if (oper == "*" or oper == "/" or oper == "-") 
      end
      return s
    end
      
    def to_var
      op1 = nil
      op1 = @operand1.to_var if @operand1.respond_to?(:to_var)
      op2 = nil
      op2 = @operand2.to_var if @operand2.respond_to?(:to_var)
      if op1 and op2 then
        r_t, oper = BOAST::transition(op1, op2, @operator)
        res_exp = BOAST::Expression::to_s_base(op1, op2, oper, r_t)
        return r_t.copy(res_exp, :const => nil, :constant => nil, :direction => nil, :dir => nil)
      elsif op2
        res_exp = BOAST::Expression::to_s_base(@operand1, op2, @operator)
        return op2.copy(res_exp, :const => nil, :constant => nil, :direction => nil, :dir => nil)
      elsif op1
        res_exp = BOAST::Expression::to_s_base(op1, @operand2, @operator)
        return op1.copy(res_exp, :const => nil, :constant => nil, :direction => nil, :dir => nil)
      else
        STDERR.puts "#{@operand1} #{@operand2}"
        raise "Expression on no operand!"
      end
    end
 
    def to_s
      op1 = nil
      op1 = @operand1.to_var if @operand1.respond_to?(:to_var)
      op2 = nil
      op2 = @operand2.to_var if @operand2.respond_to?(:to_var)
      r_t = nil
      if op1 and op2 then
        r_t, oper = BOAST::transition(op1, op2, @operator)
      else
        oper = @operator
      end

      op1 = @operand1 if op1.nil?
      op2 = @operand2 if op2.nil?

      return BOAST::Expression::to_s_base(op1, op2, oper, r_t)
    end

    def pr
      s=""
      s += BOAST::indent
      s += to_s
      s += ";" if [C, CL, CUDA].include?( BOAST::lang ) 
      BOAST::output.puts s
      return self
    end
  end

end
