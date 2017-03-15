module BOAST

  # @private
  module FORTRANRuntime
    include CompiledRuntime

    private

    def method_name
      return  "#{@procedure.name}_"
    end

    def line_limited_source
      s = ""
      @code.rewind
      @code.each_line { |line|
        if line.match(/^\s*!\w*?\$/) then
          if line.match(/^\s*!\$(omp|OMP)/) then
            chunks = line.scan(/.{1,#{fortran_line_length-7}}/)
            s += chunks.join("&\n!$omp&") + "\n"
          elsif line.match(/^\s*!(DIR|dir)\$/) then
            chunks = line.scan(/.{1,#{fortran_line_length-7}}/)
            s += chunks.join("&\n!DIR$&") + "\n"
          else
            chunks = line.scan(/.{1,#{fortran_line_length-4}}/)
            s += chunks.join("&\n!$&") + "\n"
          end
        elsif line.match(/^\s*!/) or line.match(/^\s*#include/) then
          s += line
        else
          chunks = line.scan(/.{1,#{fortran_line_length-2}}/)
          s += chunks.join("&\n&") + "\n"
        end
      }
      return s
    end

    def fill_library_source
      if fortran_line_length == 0 then
        @code.rewind
        get_output.write @code.read
      else
        get_output.print line_limited_source
      end
    end

    def create_procedure_call_parameters
      return @procedure.parameters.collect { |param|
        par = param_struct.struct_reference(param_struct.type.members[param.name.to_s])
        if param.dimension then
          "#{par}"
        else
          "&#{par}"
        end
      }
    end

  end

end
