module BOAST

  module Functors
  end

  module VarFunctors
  end

  extend Functors

  EXTENDED.push Functors

  extend VarFunctors

  EXTENDED.push VarFunctors

  module_function

  # Creates a wrapper for $1 new method
  # @param [Class] klass class to turn into a functor.
  # @!macro [attach] functorize
  #   @!method $1
  #   @scope class
  #   Creates a new $1 object, arguments are passed to the *new* method of $1
  def functorize(klass)
    name = klass.name.split('::').last
    s = <<EOF
  module Functors
    def #{name}(*args,&block)
      #{name}::new(*args,&block)
    end
  end
EOF
    eval s
  end

  # Creates a functor to create a Variable of type $1
  # @param [DataType] klass DataType to turn into a functor.
  # @!macro [attach] var_functorize
  #   @!method $1(name, *args, &block)
  #   @scope class
  #   Creates a new Variable of type $1.
  #   @param [#to_s] name name of the Variable
  #   @param [Object] args parameters to use when creating a Variable
  #   @param [Block] block block of code will be forwarded
  def var_functorize(klass)
    name = klass.name.split('::').last
    s = <<EOF
  module VarFunctors
    def #{name}(*args,&block)
      Variable::new(args[0],#{name},*args[1..-1], &block)
    end
  end
EOF
    eval s
  end

  module Functor

    def self.extended(mod)
      eval "#{mod.name.split('::')[-2]}::functorize(mod)"
    end

  end

  module VarFunctor

    def self.extended(mod)
      eval "#{mod.name.split('::')[-2]}::var_functorize(mod)"
    end

  end

end
