module BOAST

  class ControlStructure
    include PrivateStateAccessor
    include Inspectable

    def self.inherited(child)
      child.extend Functor
    end

    def self.token_string_generator(name, *args)
       s = <<EOF
    def #{name}_string(#{args.join(",")})
      return eval @@strings[get_lang][:#{name}] 
    end
EOF
    end

  end

end
