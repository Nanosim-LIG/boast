module BOAST
  module Arithmetic

    def ===(x)
      return Expression::new(Affectation,self,x)
    end

    def !
      return Expression::new(Not,nil,self)
    end

    def ==(x)
      return Expression::new("==",self,x)
    end

    def !=(x)
      return Expression::new(Different,self,x)
    end

    def >(x)
      return Expression::new(">",self,x)
    end
 
    def <(x)
      return Expression::new("<",self,x)
    end
 
    def >=(x)
      return Expression::new(">=",self,x)
    end
 
    def <=(x)
      return Expression::new("<=",self,x)
    end
 
    def +(x)
      return Expression::new(Addition,self,x)
    end

    def -(x)
      return Expression::new(Substraction,self,x)
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

    def address
      return Expression::new("&",nil,self)
    end
   
    def dereference
      return Expression::new("*",nil,self)
    end

    def and(x)
      return Expression::new("&&", self, x)
    end

    def or(x)
      return Expression::new("||", self, x)
    end

    def components( range )
      existing_set = [*('0'..'9'),*('a'..'z')].first(self.type.vector_length)
      eval "self.s#{existing_set[range].join("")}"
    end

  end
end
