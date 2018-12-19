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
        elsif line.length <= fortran_line_length-7 or line.match(/^\s*!/) or line.match(/^\s*#include/) then
          s += line
        else
          chunks = line.scan(/.{1,#{fortran_line_length-2}}/)
          sep = "&\n&"
          chunks.each_with_index{|chunk,i|
            if chunk.include?("!") then
              sep = ""
            end
            s += chunk
            s += sep if i < chunks.size - 1
          }
          s += "\n"
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

    def copy_array_param_from_ruby(str_par, param, ruby_param )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      (rb_ptr === ruby_param).pr
      get_output.print <<EOF
  if (TYPE(_boast_rb_ptr) == T_STRING) {
    #{str_par} = (void *)RSTRING_PTR(_boast_rb_ptr);
  } else if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    #{str_par} = (void *) _boast_n_ary->ptr;
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for #{param}, expecting array!");
  }
EOF
    end


    def collect_kernel_params
      pars = @procedure.parameters.collect { |param|
        if (param.vector? || param.vector_scalar?) && !param.dimension? then
          param.copy(param.name, :const => nil, :constant => nil, :dir => nil, :direction => nil, :reference => nil, :dim => Dim() )
        else
          param.copy(param.name, :const => nil, :constant => nil, :dir => nil, :direction => nil, :reference => nil )
        end
      }
      pars.push @procedure.properties[:return].copy("_boast_ret") if @procedure.properties[:return]
      pars
    end

    def create_procedure_indirect_call_parameters
      return @procedure.parameters.collect { |param|
        par = "#{param_struct.struct_reference(param_struct.type.members[param.name.to_s])}".gsub("_boast_params.","_boast_params->")
        if param.dimension || param.vector? || param.vector_scalar? then
          "#{par}"
        else
          "&#{par}"
        end
      }
    end

    def create_procedure_call_parameters
      return @procedure.parameters.collect { |param|
        par = param_struct.struct_reference(param_struct.type.members[param.name.to_s])
        if param.dimension || param.vector? || param.vector_scalar? then
          "#{par}"
        else
          "&#{par}"
        end
      }
    end

  end

end
