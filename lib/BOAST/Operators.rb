module BOAST
  class Operator
  end

  class Affectation < Operator
    def Affectation.to_s(arg1, arg2)
      return "#{arg1} = #{arg2}"
    end
  end

  class Multiplication < Operator
    def Multiplication.to_s(arg1, arg2)
      return "(#{arg1}) * (#{arg2})"
    end
  end

  class Addition < Operator
    def Addition.to_s(arg1, arg2)
      return "#{arg1} + #{arg2}"
    end
  end

  class Substraction < Operator
    def Substraction.to_s(arg1, arg2)
      return "#{arg1} - (#{arg2})"
    end
  end

  class Division < Operator
    def Division.to_s(arg1, arg2)
      return "(#{arg1}) / (#{arg2})"
    end
  end

  class Minus < Operator
    def Minus.to_s(arg1, arg2)
      return " -(#{arg2})"
    end
  end

  class Not < Operator
    def Not.to_s(arg1, arg2)
      return " ! #{arg2}"
    end
  end

end
