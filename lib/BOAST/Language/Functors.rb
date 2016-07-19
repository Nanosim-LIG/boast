module BOAST

  module Functors
  end

  module VarFunctors
  end

  extend Functors

  EXTENDED.push Functors

  extend VarFunctors

  EXTENDED.push VarFunctors


  module Functors

    module_function

    # Creates a wrapper for $1 new method
    # @param [Class] klass class to turn into a functor.
    # @!macro [attach] functorize
    #   @!method $1
    #   Creates a new $1 object, arguments are passed to the *new* method of $1. (see {$1#initialize}).
    def functorize(klass)
      name = klass.name.split('::').last
      s = <<EOF
      def #{name}(*args,&block)
        #{name}::new(*args,&block)
      end
EOF
      class_eval s
    end

  end

  module VarFunctors 

    module_function

    # Creates a functor to create a Variable of type $1
    # @param [DataType] klass DataType to turn into a functor and add it to the VarFunctors module.
    # @!macro [attach] var_functorize
    #   @!method $1(name, *args, &block)
    #   Creates a new Variable of type $1.
    #   @param [#to_s] name name of the Variable
    #   @param [Object] args parameters to use when creating a Variable
    #   @param [Block] block block of code will be forwarded
    def var_functorize(klass)
      name = klass.name.split('::').last
      s = <<EOF
      def #{name}(*args,&block)
        Variable::new(args[0],#{name},*args[1..-1], &block)
      end
EOF
      class_eval s
    end

  end

  module Functor

    def self.extended(mod)
      BOAST::Functors::functorize(mod)
    end

  end

  module VarFunctor

    def self.extended(mod)
      BOAST::VarFunctors::var_functorize(mod)
    end

  end

end
