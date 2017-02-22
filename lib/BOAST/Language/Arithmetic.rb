module BOAST

  module TopLevelExpressions

    # Creates a return Expression
    # @param [#to_var] value to return
    # @return [Expression]
    def Return(value)
      return Expression::new("return",nil, value ? value : "" )
    end

    # Creates an Expression using the boolean And Operator
    # @param [#to_var] a
    # @param [#to_var] b
    # @return [Expression]
    def And(a, b)
      return Expression::new(And, a, b)
    end

    # Creates an Expression using the boolean Or Operator
    # @param [#to_var] a
    # @param [#to_var] b
    # @return [Expression]
    def Or(a, b)
      return Expression::new(Or, a, b)
    end

    def Max(a, b)
      return Expression::new(Max, a, b)
    end

    def Min(a, b)
      return Expression::new(Min, a, b)
    end

  end

  extend TopLevelExpressions

  EXTENDED.push TopLevelExpressions

  # Defines arithmetic operation, mostly using operator overloading.
  module Arithmetic

    # Returns an Exponentiation Expression bewtween self and x
    # @param [#to_var] x
    # @return [Expression]
    def **(x)
      return Expression::new(Exponentiation,self,x)
    end

    # Returns an Affectation Expression x into self
    # @param [#to_var] x
    # @return [Expression]
    def ===(x)
      return Affectation::new(self,x)
    end

    def !
      return Expression::new(Not,nil,self)
    end

    def ==(x)
      return Expression::new(Equal,self,x)
    end

    def !=(x)
      return Expression::new(Different,self,x)
    end

    def >(x)
      return Expression::new(Greater,self,x)
    end
 
    def <(x)
      return Expression::new(Less,self,x)
    end
 
    def >=(x)
      return Expression::new(GreaterOrEqual,self,x)
    end
 
    def <=(x)
      return Expression::new(LessOrEqual,self,x)
    end
 
    def +(x)
      return Expression::new(Addition,self,x)
    end

    def -(x)
      return Expression::new(Subtraction,self,x)
    end
 
    def *(x)
      return Expression::new(Multiplication,self,x)
    end

    def /(x)
      return Expression::new(Division,self,x)
    end
 
    def -@
      return Expression::new(Minus,nil,self)
    end

    def +@
      return Expression::new(Plus,nil,self)
    end

    def reference
      return Expression::new(Reference,nil,self)
    end

    alias address reference
   
    def dereference
      return Index::new(self, *(self.dimension.collect(&:start))) if lang == FORTRAN
      return Expression::new(Dereference,nil,self)
    end

    def and(x)
      return Expression::new(And, self, x)
    end

    alias & and

    def or(x)
      return Expression::new(Or, self, x)
    end

    alias | or

    def cast(type)
      return type.copy("(#{type.type.decl}#{type.dimension? ? " *" : ""})#{self}")
    end

    def components( range )
      var = self.to_var
      if var.type.vector_length == 1
        return self
      else
        existing_set = [*('0'..'9'),*('a'..'z')].first(var.type.vector_length)
        if range.kind_of?(Range) then
          eval "self.s#{existing_set[range].join("")}"
        elsif range.kind_of?(Array) then
          eval "self.s#{existing_set.values_at(*range).join("")}"
        else
          eval "self.s#{existing_set[range]}"
        end
      end
    end

    def coerce(other)
      return [other.to_var, self]
    end

  end
end
