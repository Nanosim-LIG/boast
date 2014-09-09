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
