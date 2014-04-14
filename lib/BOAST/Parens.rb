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

    return BOAST::FuncCall::new(m,*a,&b) if s == BOAST

    orig_method_missing m, *a, &b
  end
end

