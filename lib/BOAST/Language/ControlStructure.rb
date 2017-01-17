module BOAST

  class ControlStructure
    include PrivateStateAccessor
    include Inspectable

    attr_accessor :args

    def self.inherited(child)
      child.extend Functor
    end

    def get_strings
      return { C => get_c_strings,
               CL => get_cl_strings,
               CUDA => get_cuda_strings,
               FORTRAN => get_fortran_strings }
    end

    private :get_strings

    class << self

      private

      def token_string_generator(name, *args)
       return <<EOF
    def #{name}_string(#{args.join(",")})
      return eval get_strings[get_lang][:#{name}]
    end
EOF
      end

    end

    def [](*args)
      @args = args
      return self
    end

    def initialize
      @args = nil
    end

  end

end
