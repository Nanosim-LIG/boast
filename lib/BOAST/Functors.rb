module BOAST

  module Functor

    def parens(*args,&block)
      return self::new(*args,&block)
    end

  end

  module VarFunctor

    def parens(*args,&block)
      return Variable::new(args[0], self, *args[1..-1], &block)
    end

  end

end
