module BOAST

  module_function

  FUNCCALLS = {}

  def register_funccall(name, options = {})
    sym = name.to_sym
    FUNCCALLS[sym] = {}
    FUNCCALLS[sym][:parameters] = options[:parameters]
    FUNCCALLS[sym][:returns] = options[:returns]
    s =<<EOF
    def self.#{name}(*args)
      return FuncCall(#{sym.inspect}, *args#{options[:returns] ? ", returns: FUNCCALLS[#{sym.inspect}][:returns]" : ""})
    end
EOF
    eval s
  end

end
