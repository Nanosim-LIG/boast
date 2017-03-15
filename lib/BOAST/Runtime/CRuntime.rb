module BOAST

  # @private
  module CRuntime
    include CompiledRuntime

    private

    def fill_library_header
      get_output.puts "#include <stdlib.h>"
      get_output.puts "#include <inttypes.h>"
      get_output.puts "#include <math.h>"
      @includes.each { |inc|
        get_output.puts "#include \"#{inc}\""
      }
      get_output.puts "#define __assume_aligned(lvalueptr, align) lvalueptr = __builtin_assume_aligned (lvalueptr, align)" if @compiler_options[:CC].match("gcc")
    end

    def fill_library_source
      fill_library_header
      @kernels.each { |k|
        k.procedure.boast_header
      }
      @code.rewind
      get_output.write @code.read
    end

    def create_procedure_call_parameters
      return @procedure.parameters.collect { |param|
        par = param_struct.struct_reference(param_struct.type.members[param.name.to_s])
        if param.dimension then
          "#{par}"
        elsif param.direction == :out or param.direction == :inout or param.reference? then
          "&#{par}"
        else
          "#{par}"
        end
      }
    end

  end

end
