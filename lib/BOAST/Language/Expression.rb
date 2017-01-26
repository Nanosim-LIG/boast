module BOAST

  module_function

  # @!parse module Functors; functorize Expression; end
  class Expression
    include PrivateStateAccessor
    include Arithmetic
    include Inspectable
    extend Functor
    include TypeTransition
    include Annotation

    ANNOTATIONS = [:operator, :operand1, :operand2]

    def method_missing(m, *a, &b)
      var = to_var
      if var.type.methods.include?(:members) and var.type.members[m.to_s] then
        return struct_reference(type.members[m.to_s])
      elsif var.vector? and m.to_s[0] == 's' and lang != CUDA then
        required_set = m.to_s[1..-1].chars.to_a
        existing_set = [*('0'..'9'),*('a'..'z')].first(var.type.vector_length)
        if required_set.length == required_set.uniq.length and (required_set - existing_set).empty? then
          return var.copy(var.name+"."+m.to_s, :vector_length => m.to_s[1..-1].length) if lang == CL
          return var.copy("(#{var.name})(#{existing_set.index(required_set[0])+1})", :vector_length => 1) if lang == FORTRAN
          return var.copy("(#{var.name})[#{existing_set.index(required_set[0])}]", :vector_length => 1) if lang == C and architecture == X86
          return super
        else
          return super
        end
      else
        return super
      end
    end

    attr_reader :operator
    attr_reader :operand1
    attr_reader :operand2
    def initialize(operator, operand1, operand2)
      @operator = operator
      @operand1 = operand1
      @operand2 = operand2
      if @operand1.nil? and @operand2.nil?
        STDERR.puts "#{@operand1} #{@operand2}"
        raise "Expression on no operand!"
      end
    end

    def to_s_base(op1, op2, oper, return_type = nil)
      return oper.string(op1, op2, return_type) unless oper.kind_of?(String)
      s = ""
      if op1 then
        s += "(" if (oper == "*" or oper == "/") 
        s += op1.to_s
        s += ")" if (oper == "*" or oper == "/") 
      end        
      s += " " unless oper == "++" or oper == "."
      s += oper unless ( oper == "&" and lang == FORTRAN )
      s += " " unless oper == "." or oper == "&" or ( oper == "*" and op1.nil? )
      if op2 then
        s += "(" if (oper == "*" or oper == "/" or oper == "-") 
        s += op2.to_s
        s += ")" if (oper == "*" or oper == "/" or oper == "-") 
      end
      return s
    end

    private :to_s_base
      
    def to_var
      op1 = nil
      op1 = @operand1.to_var if @operand1.respond_to?(:to_var)
      op2 = nil
      op2 = @operand2.to_var if @operand2.respond_to?(:to_var)
      if op1 and op2 then
        r_t, oper = transition(op1, op2, @operator)
        res_exp = to_s_base(op1, op2, oper, r_t)
        return r_t.copy(res_exp, :const => nil, :constant => nil, :direction => nil, :dir => nil, :align => nil)
      elsif op2
        res_exp = to_s_base(@operand1, op2, @operator)
        return op2.copy(res_exp, :const => nil, :constant => nil, :direction => nil, :dir => nil, :align => nil)
      elsif op1
        res_exp = to_s_base(op1, @operand2, @operator)
        return op1.copy(res_exp, :const => nil, :constant => nil, :direction => nil, :dir => nil, :align => nil)
      else
        res_exp = to_s_base(@operand1, @operand2, @operator)
        return Variable::new(res_exp, get_default_type)
      end
    end
 
    def to_s
      op1 = nil
      op1 = @operand1.to_var if @operand1.respond_to?(:to_var)
      op2 = nil
      op2 = @operand2.to_var if @operand2.respond_to?(:to_var)
      r_t = nil
      if op1 and op2 then
        r_t, oper = transition(op1, op2, @operator)
      else
        oper = @operator
      end

      op1 = @operand1 if op1.nil?
      op2 = @operand2 if op2.nil?

      return to_s_base(op1, op2, oper, r_t)
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
