module BOAST
  
  module_function
  def functorize(klass)
    name = klass.name.split('::').last
    s = <<EOF
  def #{name}(*args,&block)
     #{name}::new(*args,&block)
  end

  module_function :#{name}
EOF
    eval s
  end

  def var_functorize(klass)
    name = klass.name.split('::').last
    s = <<EOF
  def #{name}(*args,&block)
     Variable::new(args[0],#{name},*args[1..-1], &block)
  end

  module_function :#{name}
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
