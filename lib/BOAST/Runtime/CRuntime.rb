module BOAST

  # @private
  module CRuntime
    include CompiledRuntime

    private

    def define_globals
      @globals.each { |g|
        get_output.print "extern "
        decl g
      }
    end

    def copy_array_global_from_ruby(str_par, param, ruby_param)
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      (rb_ptr === ruby_param).pr
      get_output.print <<EOF
  if (TYPE(_boast_rb_ptr) == T_STRING) {
    #{
  if param.dimension then
    "memcpy(#{str_par}, (void *)RSTRING_PTR(_boast_rb_ptr), RSTRING_LEN(_boast_rb_ptr))"
  elsif param.vector_scalar? then
    (str_par === param.copy("*(#{param.type.decl} *)RSTRING_PTR(_boast_rb_ptr)", :dimension => BOAST.Dim(), :vector_length => nil)).to_s
  else
    (str_par === param.copy("*(void *)RSTRING_PTR(_boast_rb_ptr)", :dimension => BOAST.Dim(), :vector_length => nil)).to_s
  end
    };
  } else if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    size_t _boast_array_size;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    _boast_array_size = _boast_n_ary->total * na_sizeof[_boast_n_ary->type];
    #{
  if param.dimension then
    "memcpy(#{str_par}, (void *) _boast_n_ary->ptr, _boast_array_size)"
  elsif param.vector_scalar? then
    (str_par === param.copy("*(#{param.type.decl} *) _boast_n_ary->ptr", :dimension => BOAST.Dim(), :vector_length => nil)).to_s
  else
    (str_par === param.copy("*(void *) _boast_n_ary->ptr", :dimension => BOAST.Dim(), :vector_length => nil)).to_s
  end
    };
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for #{param}, expecting NArray or String!");
  }
EOF
    end

    def create_set_globals
      get_output.print <<EOF
static VALUE method_set_globals(int _boast_argc, VALUE *_boast_argv, VALUE _boast_self) {
  VALUE _boast_rb_ptr = Qnil;
EOF
      argc = @globals.size
      argv = Variable::new("_boast_argv", CustomType, :type_name => "VALUE", :dimension => [ Dimension::new(0,argc-1) ] )
      push_env(:decl_module => true, :indent_level => 2) {
      @globals.each_with_index { |param, i|
        if param.dimension? or param.vector? or param.vector_scalar? then
          copy_array_global_from_ruby(param, param, argv[i])
        else
          copy_scalar_param_from_ruby(param, param, argv[i])
        end
      }
      get_output.print <<EOF
  return Qnil;
}
EOF
      }
    end

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

    def create_procedure_indirect_call_parameters
      return @procedure.parameters.collect { |param|
        par = "#{param_struct.struct_reference(param_struct.type.members[param.name.to_s])}".gsub("_boast_params.","_boast_params->")
        if param.dimension then
          "#{par}"
        elsif param.direction == :out or param.direction == :inout or param.reference? then
          "&#{par}"
        else
          "#{par}"
        end
      }
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
