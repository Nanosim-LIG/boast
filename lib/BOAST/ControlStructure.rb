module BOAST

  class ControlStructure

    def self.token_string_generator(name, *args)
       s = <<EOF
    def #{name}_string(#{args.join(",")})
      return eval @@strings[BOAST::get_lang][:#{name}] 
    end
EOF
    end

  end

end
