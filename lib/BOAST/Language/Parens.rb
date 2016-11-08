module BOAST

  module_function

  FUNCCALLS = {}

  def register_funccall(name, options = {})
    sym = name.to_sym
    ret = options[:return] ? options[:return] : options[:returns]
    FUNCCALLS[sym] = {}
    FUNCCALLS[sym][:parameters] = options[:parameters]
    FUNCCALLS[sym][:return] = ret
    s =<<EOF
    def self.#{name}(*args)
      return FuncCall(#{sym.inspect}, *args#{ ret ? ", return: FUNCCALLS[#{sym.inspect}][:return]" : ""})
    end
EOF
    eval s
  end

end
