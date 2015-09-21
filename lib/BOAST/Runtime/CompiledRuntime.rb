require 'stringio'
require 'rake'
require 'tempfile'
require 'rbconfig'
require 'systemu'
require 'pathname'
require 'os'

class Dir
  module Tmpname
    module_function
    def make_tmpname(prefix_suffix, n)
      case prefix_suffix
      when String
        prefix = prefix_suffix
        suffix = ""
      when Array
        prefix = prefix_suffix[0]
        suffix = prefix_suffix[1]
      else
        raise ArgumentError, "unexpected prefix_suffix: #{prefix_suffix.inspect}"
      end
      t = Time.now.strftime("%Y%m%d")
      path = "#{prefix}#{t}_#{$$}_#{rand(0x100000000).to_s(36)}"
      path << "_#{n}" if n
      path << suffix
    end
  end
end

module BOAST

  module CompiledRuntime
    attr_accessor :binary
    attr_accessor :source

    @@extensions = {
      C => ".c",
      CUDA => ".cu",
      FORTRAN => ".f90"
    }

    def base_name
      return File::split(@marker.path)[1]
    end

    def module_name
      return "Mod_" + base_name#.gsub("-","_")
    end

    def directory
      return File::split(@marker.path)[0]
    end

    def module_file_base_name
      return "Mod_" + base_name
    end

    def module_file_base_path
      return "#{directory}/#{module_file_base_name}"
    end

    def module_file_path
      return "#{module_file_base_path}.#{RbConfig::CONFIG["DLEXT"]}"
    end

    def module_file_source
      return module_file_base_path + ".c"
    end

    def module_file_object
      return "#{module_file_base_path}.#{RbConfig::CONFIG["OBJEXT"]}"
    end

    def base_path
      return "#{directory}/#{base_name}"
    end

    def library_source
      return base_path + @@extensions[@lang]
    end

    def library_object
      return "#{base_path}.#{RbConfig::CONFIG["OBJEXT"]}"
    end

    def library_path
      return "#{base_path}.#{RbConfig::CONFIG["DLEXT"]}"
    end

    def target
      return module_file_path
    end

    def target_depends
      return [ module_file_object, library_object ]
    end

    def save_binary
      f = File::open(library_object,"rb")
      @binary = StringIO::new
      @binary.write( f.read )
      f.close
    end

    def save_source
      f = File::open(library_source,"r")
      @source = StringIO::new
      @source.write( f.read )
      f.close
    end

    def create_targets( linker, ldshared, ldflags, kernel_files)
      file target => target_depends do
        #puts "#{linker} #{ldshared} -o #{target} #{target_depends.join(" ")} #{(kernel_files.collect {|f| f.path}).join(" ")} #{ldflags}"
        sh "#{linker} #{ldshared} -o #{target} #{target_depends.join(" ")} #{(kernel_files.collect {|f| f.path}).join(" ")} #{ldflags}"
      end
      Rake::Task[target].invoke
    end

    def method_name
      return @procedure.name
    end

    def get_sub_kernels
      kernel_files = []
      @kernels.each { |kernel|
        kernel_file = Tempfile::new([kernel.procedure.name,".#{RbConfig::CONFIG["OBJEXT"]}"])
        kernel.binary.rewind
        kernel_file.write( kernel.binary.read )
        kernel_file.close
        kernel_files.push(kernel_file)
      }
      return kernel_files
    end

    def create_library_source
      f = File::open(library_source,"w+")
      previous_lang = get_lang
      previous_output = get_output
      set_output(f)
      set_lang(@lang)

      fill_library_source

      if debug_source? then
        f.rewind
        puts f.read
      end
      set_output(previous_output)
      set_lang(previous_lang)
      f.close
    end

    def fill_module_header
      get_output.print <<EOF
#include "ruby.h"
#include <inttypes.h>
#ifdef HAVE_NARRAY_H
#include "narray.h"
#endif
EOF
    end

    def fill_module_preamble
      get_output.print <<EOF
VALUE #{module_name} = Qnil;
void Init_#{module_name}();
VALUE method_run(int _boast_argc, VALUE *_boast_argv, VALUE _boast_self);
void Init_#{module_name}() {
  #{module_name} = rb_define_module("#{module_name}");
  rb_define_method(#{module_name}, "run", method_run, -1);
}
EOF
    end

    def fill_check_args
      get_output.print <<EOF
  VALUE _boast_rb_opts;
  if( _boast_argc < #{@procedure.parameters.length} || _boast_argc > #{@procedure.parameters.length + 1} )
    rb_raise(rb_eArgError, "Wrong number of arguments for #{@procedure.name} (%d for #{@procedure.parameters.length})!", _boast_argc);
  _boast_rb_opts = Qnil;
  if( _boast_argc == #{@procedure.parameters.length + 1} ) {
    _boast_rb_opts = _boast_argv[_boast_argc -1];
    if ( _boast_rb_opts != Qnil ) {
      if (TYPE(_boast_rb_opts) != T_HASH)
        rb_raise(rb_eArgError, "Options should be passed as a hash");
    }
  }
EOF
    end

    def fill_decl_module_params
      set_decl_module(true)
      @procedure.parameters.each { |param|
        param_copy = param.copy
        param_copy.constant = nil
        param_copy.direction = nil
        param_copy.decl
      }
      get_output.puts "  #{@procedure.properties[:return].type.decl} _boast_ret;" if @procedure.properties[:return]
      get_output.puts "  VALUE _boast_stats = rb_hash_new();"
      get_output.puts "  VALUE _boast_rb_ptr = Qnil;"
      refs = false
      @procedure.parameters.each_with_index do |param,i|
        refs = true if param.scalar_output?
      end
      if refs then
        get_output.puts "  VALUE _boast_refs = rb_hash_new();"
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"reference_return\")),_boast_refs);"
      end
      set_decl_module(false)
    end

    def copy_scalar_param_from_ruby( param, ruby_param )
      case param.type
      when Int
        (param === FuncCall::new("NUM2INT", ruby_param)).pr if param.type.size == 4
        (param === FuncCall::new("NUM2LONG", ruby_param)).pr if param.type.size == 8
      when Real
        (param === FuncCall::new("NUM2DBL", ruby_param)).pr
      end
    end

    def copy_array_param_from_ruby( param, ruby_param )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      (rb_ptr === ruby_param).pr
      get_output.print <<EOF
  if (TYPE(_boast_rb_ptr) == T_STRING) {
    #{param} = (void *) RSTRING_PTR(_boast_rb_ptr);
  } else if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    #{param} = (void *) _boast_n_ary->ptr;
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for %s, expecting array!", "#{param}");
  }
EOF
    end

    def get_params_value
      argc = @procedure.parameters.length
      argv = Variable::new("_boast_argv", CustomType, :type_name => "VALUE", :dimension => [ Dimension::new(0,argc-1) ] )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      set_decl_module(true)
      @procedure.parameters.each_index do |i|
        param = @procedure.parameters[i]
        if not param.dimension then
          copy_scalar_param_from_ruby(param, argv[i])
        else
          copy_array_param_from_ruby(param, argv[i])
        end
      end
      set_decl_module(false)
    end

    def create_procedure_call
      get_output.print "  _boast_ret = " if @procedure.properties[:return]
      get_output.print " #{method_name}( "
      get_output.print create_procedure_call_parameters.join(", ")
      get_output.puts " );"
    end

    def copy_scalar_param_to_ruby(param, ruby_param)
      if param.scalar_output? then
        case param.type
        when Int
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_int_new((long long)#{param}));" if param.type.signed?
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_int_new((unsigned long long)#{param}));" if not param.type.signed?
        when Real
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_float_new((double)#{param}));"
        end
      end
    end

    def copy_array_param_to_ruby(param, ruby_param)
    end

    def get_results
      argc = @procedure.parameters.length
      argv = Variable::new("_boast_argv", CustomType, :type_name => "VALUE", :dimension => [ Dimension::new(0,argc-1) ] )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      set_decl_module(true)
      @procedure.parameters.each_index do |i|
        param = @procedure.parameters[i]
        if not param.dimension then
          copy_scalar_param_to_ruby(param, argv[i])
        else
          copy_array_param_to_ruby(param, argv[i])
        end
      end
      set_decl_module(false)
    end

    def store_results
      if @procedure.properties[:return] then
        type_ret = @procedure.properties[:return].type
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_int_new((long long)_boast_ret));" if type_ret.kind_of?(Int) and type_ret.signed
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_int_new((unsigned long long)_boast_ret));" if type_ret.kind_of?(Int) and not type_ret.signed
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_float_new((double)_boast_ret));" if type_ret.kind_of?(Real)
      end
    end

    def fill_module_file_source
      fill_module_header
      @probes.map(&:header)
      @procedure.boast_header(@lang)

      fill_module_preamble

      set_transition("VALUE", "VALUE", :default,  CustomType::new(:type_name => "VALUE"))
      get_output.puts "VALUE method_run(int _boast_argc, VALUE *_boast_argv, VALUE _boast_self) {"
      increment_indent_level

      fill_check_args

      fill_decl_module_params

      @probes.reverse.map(&:decl)

      get_params_value

      @probes.map(&:configure)

      @probes.reverse.map(&:start)

      create_procedure_call

      @probes.map(&:stop)

      get_results

      @probes.map(&:compute)

      store_results

      get_output.puts "  return _boast_stats;"
      decrement_indent_level
      get_output.puts "}"
    end

    def create_module_file_source
      f = File::open(module_file_source, "w+")
      previous_lang = get_lang
      previous_output = get_output
      set_output(f)
      set_lang(C)

      fill_module_file_source

      if debug_source? then
        f.rewind
        puts f.read
      end
      set_output(previous_output)
      set_lang(previous_lang)
      f.close
    end

    def create_sources
      create_library_source
      create_module_file_source
    end

    def load_module
      require module_file_path
    end

    def target_sources
      return [ module_file_source, library_source ]
    end

    def cleanup(kernel_files)
      ([target] + target_depends + target_sources).each { |fn|
        File::unlink(fn)
      }
      kernel_files.each { |f|
        f.unlink
      }
    end

    def build(options={})
      compiler_options = BOAST::get_compiler_options
      compiler_options.update(options)
      linker, ldshared, ldflags = setup_compilers(compiler_options)
      @compiler_options = compiler_options

      @marker = Tempfile::new([@procedure.name,""])

      kernel_files = get_sub_kernels

      create_sources

      save_source

      create_targets(linker, ldshared, ldflags, kernel_files)

      save_binary

      load_module

      cleanup(kernel_files)

      eval "self.extend(#{module_name})"

      return self
    end

    def dump_binary
      f = File::open(library_object,"wb")
      @binary.rewind
      f.write( @binary.read )
      f.close
    end

    def dump_source
      f = File::open(library_source,"wb")
      @source.rewind
      f.write( @source.read )
      f.close
    end

  end

end
