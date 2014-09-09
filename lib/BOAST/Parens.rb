class Object
  alias_method :orig_method_missing, :method_missing
  
  def method_missing(m, *a, &b)
    #puts "WARNING: unknown #{m}"
    s=nil
    klass = begin
      s = (self.is_a?(Module) ? self : self.class)
      s.const_get(m)
    rescue NameError
    end
    
    return klass.send(:parens, *a, &b)  if klass.respond_to? :parens

    if s == BOAST then
      STDERR.puts "Warning unkwown function #{m} generated as BOAST::FuncCall!" if BOAST::debug
      return BOAST::FuncCall::new(m,*a,&b)
    end

    orig_method_missing m, *a, &b
  end

end

module BOAST

  module_function

  def register_funccall(name)
    s =<<EOF
    def self.#{name}(*args)
      return BOAST::FuncCall("#{name}", *args)
    end
EOF
    eval s
  end

end
