module BOAST

  # @private
  module FORTRANRuntime
    include CompiledRuntime

    private

    def method_name
      return @procedure.name + "_"
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
        elsif line.match(/^\s*!/) then
          s += line
        elsif not line.match(/^\s*(include|INCLUDE)/) then
          chunks = line.scan(/.{1,#{fortran_line_length-2}}/)
          s += chunks.join("&\n&") + "\n"
        end
      }
      return s
    end

    def fill_library_source
      get_output.print line_limited_source
    end

    def create_procedure_call_parameters
      params = []
      @procedure.parameters.each { |param|
        if param.dimension then
          params.push( param.name )
        else
          params.push( "&"+param.name )
        end
      }
      return params
    end

  end

end
