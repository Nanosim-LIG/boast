module BOAST

  module FORTRANRuntime
    include CompiledRuntime

    def method_name
      return @procedure.name + "_"
    end

    def fill_library_source
      @code.rewind
      @code.each_line { |line|
        # check for omp pragmas
        if line.match(/^\s*!\$/) then
          if line.match(/^\s*!\$(omp|OMP)/) then
            chunks = line.scan(/.{1,#{FORTRAN_LINE_LENGTH-7}}/)
            get_output.puts chunks.join("&\n!$omp&")
          else
            chunks = line.scan(/.{1,#{FORTRAN_LINE_LENGTH-4}}/)
            get_output.puts chunks.join("&\n!$&")
          end
        elsif line.match(/^\w*!/) then
          get_output.write line
        else
          chunks = line.scan(/.{1,#{FORTRAN_LINE_LENGTH-2}}/)
          get_output.puts chunks.join("&\n&")
        end
      }
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
