module BOAST

  class ControlStructure
    include PrivateStateAccessor
    include Inspectable

    def self.inherited(child)
      child.extend Functor
    end

    def get_strings
      return { C => get_c_strings,
               CL => get_cl_strings,
               CUDA => get_cuda_strings,
               FORTRAN => get_fortran_strings }
    end

    def self.token_string_generator(name, *args)
       s = <<EOF
    def #{name}_string(#{args.join(",")})
      return eval get_strings[get_lang][:#{name}] 
    end
EOF
    end

  end

end
