module BOAST
  module Arithmetic

    def ===(x)
      return BOAST::Expression::new(BOAST::Affectation,self,x)
    end

    def !
      return BOAST::Expression::new(BOAST::Not,nil,self)
    end

    def ==(x)
      return BOAST::Expression::new("==",self,x)
    end

    def !=(x)
      return BOAST::Expression::new(BOAST::Different,self,x)
    end

    def >(x)
      return BOAST::Expression::new(">",self,x)
    end
 
    def <(x)
      return BOAST::Expression::new("<",self,x)
    end
 
    def >=(x)
      return BOAST::Expression::new(">=",self,x)
    end
 
    def <=(x)
      return BOAST::Expression::new("<=",self,x)
    end
 
    def +(x)
      return BOAST::Expression::new(BOAST::Addition,self,x)
    end

    def -(x)
      return BOAST::Expression::new(BOAST::Substraction,self,x)
    end
 
    def *(x)
      return BOAST::Expression::new(BOAST::Multiplication,self,x)
    end

    def /(x)
      return BOAST::Expression::new(BOAST::Division,self,x)
    end
 
    def -@
      return BOAST::Expression::new(BOAST::Minus,nil,self)
    end

    def address
      return BOAST::Expression::new("&",nil,self)
    end
   
    def dereference
      return BOAST::Expression::new("*",nil,self)
    end

  end
end
