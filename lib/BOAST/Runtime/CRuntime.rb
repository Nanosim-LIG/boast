module BOAST

  module CRuntime
    include CompiledRuntime

    def fill_library_header
      get_output.puts "#include <inttypes.h>"
    end

    def fill_library_source
      fill_library_header
      @code.rewind
      get_output.write code.read
    end

    def create_procedure_call_parameters
      params = []
      @procedure.parameters.each { |param|
        if param.dimension then
          params.push( param.name )
        elsif param.direction == :out or param.direction == :inout then
          params.push( "&"+param.name )
        else
          params.push( param.name )
        end
      }
      return params
    end

  end

end
