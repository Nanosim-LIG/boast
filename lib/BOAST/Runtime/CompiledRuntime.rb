require 'stringio'
require 'rake'
require 'tempfile'
require 'rbconfig'
require 'systemu'
require 'pathname'
require 'os'

class Dir
  module Tmpname
    class << self
      undef_method :make_tmpname
    end

    undef_method :make_tmpname

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

    module_function :make_tmpname

  end
end

module BOAST

  module CompiledRuntime
    attr_accessor :binary
    attr_accessor :source

    private

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

    def library_nofpic_object
      return "#{base_path}.nofpic#{RbConfig::CONFIG["OBJEXT"]}"
    end

    def library_path
      return "#{base_path}.#{RbConfig::CONFIG["DLEXT"]}"
    end

    def target
      return module_file_path
    end

    def executable_source
      return "#{base_path}_executable.c"
    end

    def executable_object
      return "#{base_path}_executable.nofpic#{RbConfig::CONFIG["OBJEXT"]}"
    end

    def target_executable
      return "#{base_path}_executable"
    end

    def target_depends
      return [ module_file_object, library_object ]
    end

    def target_executable_depends
      return [ library_nofpic_object, executable_object ]
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

    def save_module
      f = File::open(module_file_path, "rb")
      @module_binary = StringIO::new
      @module_binary.write( f.read )
      f.close
    end

    def create_targets( linker, ldshared, ldshared_flags, ldflags, kernel_files)
      file target => target_depends do
        #puts "#{linker} #{ldshared} -o #{target} #{target_depends.join(" ")} #{(kernel_files.collect {|f| f.path}).join(" ")} #{ldshared_flags} #{ldflags}"
        sh "#{linker} #{ldshared} -o #{target} #{target_depends.join(" ")} #{(kernel_files.collect {|f| f.path}).join(" ")} #{ldshared_flags} #{ldflags}"
      end
      Rake::Task[target].invoke
    end

    def create_executable_target( linker, ldflags, kernel_files)
      file target_executable => target_executable_depends do
        sh "#{linker} -o #{target_executable} #{target_executable_depends.join(" ")} #{(kernel_files.collect {|f| f.path}).join(" ")} #{ldflags}"
      end
      Rake::Task[target_executable].invoke
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
      push_env( :output => f, :lang => @lang ) {
        fill_library_source

        if debug_source? or debug_kernel_source? then
          f.rewind
          puts f.read
        end
      }
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
      @includes.each { |inc|
        get_output.puts "#include \"#{inc}\""
      }
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
  int _boast_repeat = 1;
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

    def add_run_options
      get_output.print <<EOF
  VALUE _boast_run_opts;
  _boast_run_opts = rb_const_get(rb_cObject, rb_intern("BOAST"));
  _boast_run_opts = rb_funcall(_boast_run_opts, rb_intern("get_run_config"), 0);
  if ( NUM2UINT(rb_funcall(_boast_run_opts, rb_intern("size"), 0)) > 0 ) {
    if ( _boast_rb_opts != Qnil )
      rb_funcall(_boast_run_opts, rb_intern("update"), 1, _boast_rb_opts);
    _boast_rb_opts = _boast_run_opts;
  }
  if ( _boast_rb_opts != Qnil ){
    VALUE _boast_repeat_value = Qnil;
    _boast_repeat_value = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("repeat")));
    if(_boast_repeat_value != Qnil)
      _boast_repeat = NUM2UINT(_boast_repeat_value);
    if(_boast_repeat < 0)
      _boast_repeat = 1;
  }
EOF
    end

    def fill_decl_module_params
      push_env(:decl_module => true) {
        @procedure.parameters.each { |param|
          param_copy = param.copy
          param_copy.constant = nil
          param_copy.direction = nil
          param_copy.reference = nil
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
      }
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
    #{
  if param.dimension then
    "#{param} = (void *)RSTRING_PTR(_boast_rb_ptr)"
  else
    (param === param.copy("*(void *)RSTRING_PTR(_boast_rb_ptr)", :dimension => Dim(), :vector_length => 1)).to_s
  end
    };
  } else if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    #{
  if param.dimension then
    "#{param} = (void *) _boast_n_ary->ptr"
  else
    (param === param.copy("*(void *) _boast_n_ary->ptr", :dimension => Dim(), :vector_length => 1)).to_s
  end
    };
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for %s, expecting array!", "#{param}");
  }
EOF
    end

    def get_params_value
      argc = @procedure.parameters.length
      argv = Variable::new("_boast_argv", CustomType, :type_name => "VALUE", :dimension => [ Dimension::new(0,argc-1) ] )
      push_env(:decl_module => true) {
        @procedure.parameters.each_index do |i|
          param = @procedure.parameters[i]
          if not param.dimension? and not param.vector? then
            copy_scalar_param_from_ruby(param, argv[i])
          else
            copy_array_param_from_ruby(param, argv[i])
          end
        end
      }
    end

    def create_procedure_call
      get_output.puts  "  int _boast_i;"
      get_output.puts  "  for(_boast_i = 0; _boast_i < _boast_repeat; ++_boast_i){"
      get_output.print "    _boast_ret = " if @procedure.properties[:return]
      get_output.print "    #{method_name}( "
      get_output.print create_procedure_call_parameters.join(", ")
      get_output.print "    );"
      get_output.puts  "  }"
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

    def copy_scalar_param_to_file(param)
      if param.scalar_output? then
        get_output.puts <<EOF
  __boast_f = fopen("#{@tmp_dir}/#{@procedure.name}/#{base_name}/#{param}.out", "wb");
  fwrite(&#{param}, sizeof(#{param}), 1, __boast_f);
  fclose(__boast_f);
EOF
      end
    end

    def copy_scalar_param_from_file(param)
      get_output.puts <<EOF
  __boast_f = fopen("#{@tmp_dir}/#{@procedure.name}/#{base_name}/#{param}.in", "rb");
  fread(&#{param}, sizeof(#{param}), 1, __boast_f);
  fclose(__boast_f);
EOF
    end

    def copy_array_param_to_ruby(param, ruby_param)
    end

    def copy_array_param_to_file(param)
      if param.direction == :out or param.direction == :inout then
        get_output.puts <<EOF
  __boast_f = fopen("#{@tmp_dir}/#{@procedure.name}/#{base_name}/#{param}.out", "wb");
  fwrite(#{param}, 1, __boast_sizeof_#{param}, __boast_f);
  fclose(__boast_f);
  free(#{param});
EOF
      else
        get_output.puts <<EOF
  free(#{param});
EOF
      end
    end

    def copy_array_param_from_file(param)
      get_output.puts <<EOF
  __boast_f = fopen("#{@tmp_dir}/#{@procedure.name}/#{base_name}/#{param}.in", "rb");
  fseek(__boast_f, 0L, SEEK_END);
  __boast_sizeof_#{param} = ftell(__boast_f);
  rewind(__boast_f);
  #{param} = malloc(__boast_sizeof_#{param});
  fread(#{param}, 1, __boast_sizeof_#{param}, __boast_f);
  fclose(__boast_f);
EOF
    end

    def get_results
      argc = @procedure.parameters.length
      argv = Variable::new("_boast_argv", CustomType, :type_name => "VALUE", :dimension => [ Dimension::new(0,argc-1) ] )
      push_env(:decl_module => true) {
        @procedure.parameters.each_index do |i|
          param = @procedure.parameters[i]
          if not param.dimension then
            copy_scalar_param_to_ruby(param, argv[i])
          else
            copy_array_param_to_ruby(param, argv[i])
          end
        end
      }
    end

    def store_results
      if @procedure.properties[:return] then
        type_ret = @procedure.properties[:return].type
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_int_new((long long)_boast_ret));" if type_ret.kind_of?(Int) and type_ret.signed
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_int_new((unsigned long long)_boast_ret));" if type_ret.kind_of?(Int) and not type_ret.signed
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_float_new((double)_boast_ret));" if type_ret.kind_of?(Real)
      end
    end

    def get_executable_params_value
      push_env(:decl_module => true) {
        @procedure.parameters.each do |param|
          if not param.dimension? then
            copy_scalar_param_from_file(param)
          else
            copy_array_param_from_file(param)
          end
        end
      }
    end

    def get_executable_params_return_value
      push_env(:decl_module => true) {
        @procedure.parameters.each do |param|
          if not param.dimension then
            copy_scalar_param_to_file(param)
          else
            copy_array_param_to_file(param)
          end
        end
      }
    end

    def fill_executable_source
      get_output.puts "#include <inttypes.h>"
      get_output.puts "#include <stdlib.h>"
      get_output.puts "#include <stdio.h>"
      @includes.each { |inc|
        get_output.puts "#include \"#{inc}\""
      }
      @probes.map(&:header)
      @procedure.boast_header(@lang)

      get_output.print <<EOF
void Init_#{base_name}( void );
int _boast_repeat;
void Init_#{base_name}( void ) {
EOF
      increment_indent_level
      output.puts "  FILE * __boast_f;"
      push_env(:decl_module => true) {
        @procedure.parameters.each { |param|
          if param.dimension? then
            output.puts "  size_t __boast_sizeof_#{param};"
          end
          param_copy = param.copy
          param_copy.constant = nil
          param_copy.direction = nil
          param_copy.reference = nil
          param_copy.decl
        }
        get_output.puts "  #{@procedure.properties[:return].type.decl} _boast_ret;" if @procedure.properties[:return]
      }
      @probes.reverse.map(&:decl)
      @probes.map(&:configure)

      get_executable_params_value

      @probes.reverse.map(&:start)

      get_output.puts  "  int _boast_i;"
      get_output.puts  "  for(_boast_i = 0; _boast_i < _boast_repeat; ++_boast_i){"
      get_output.print "    "
      get_output.print "_boast_ret = " if @procedure.properties[:return]
      get_output.print "#{method_name}( "
      get_output.print create_procedure_call_parameters.join(", ")
      get_output.puts  " );"
      get_output.puts  "  }"

      @probes.map(&:stop)

      get_output.puts '  printf("---\n");'
      if @procedure.properties[:return] then
        type_ret = @procedure.properties[:return].type
        get_output.puts '  printf(":return: %ld\n", (long long)_boast_ret);' if type_ret.kind_of?(Int) and type_ret.signed
        get_output.puts '  printf(":return: %uld\n", (unsigned long long)_boast_ret);' if type_ret.kind_of?(Int) and not type_ret.signed
        get_output.puts '  printf(":return: %lf\n", (double)_boast_ret);' if type_ret.kind_of?(Real)

      end

      get_executable_params_return_value

      @probes.map(&:compute)

      @probes.map(&:to_yaml)

      decrement_indent_level
      get_output.print <<EOF
}
int main(int argc, char * argv[]) {
  _boast_repeat=atoi(argv[1]);
  Init_#{base_name}();
  return 0;
}
EOF

    end

    def create_executable_source
      f = File::open(executable_source, "w+")
      push_env(:output => f, :lang => C) {
        fill_executable_source

        if debug_source? then
          f.rewind
          puts f.read
        end
      }
      f.close
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

      add_run_options

      fill_decl_module_params

      @probes.reverse.map(&:decl)

      get_params_value

      @probes.map(&:configure)

      @probes.reverse.map(&:start)

      create_procedure_call

      @probes.map(&:stop)

      get_results

      @probes.each { |p|
        p.compute
        p.store
      }

      store_results

      get_output.puts "  return _boast_stats;"
      decrement_indent_level
      get_output.puts "}"
    end

    def create_module_file_source
      f = File::open(module_file_source, "w+")
      push_env(:output => f, :lang => C) {
        fill_module_file_source

        if debug_source? then
          f.rewind
          puts f.read
        end
      }
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

    def run_executable(*params)
      options = {:repeat => 1}
      if params.last.kind_of?(Hash) then
        options.update(params.last)
        ps = params[0..-2]
      else
        ps = params[0..-1]
      end

      dump_ref_inputs( { base_name => ps }, @tmp_dir )
      boast_ret = YAML::load `#{target_executable} #{options[:repeat]}`
      res = load_ref_outputs(@tmp_dir)["#{@tmp_dir}/#{@procedure.name}/#{base_name}"]
      @procedure.parameters.each_with_index { |param, indx|
        if param.direction == :in or param.constant then
          next
        end
        if param.dimension then
          ps[indx][0..-1] = res[indx][0..-1]
        else
          boast_ret[:reference_return] = {} unless boast_ret[:reference_return]
          boast_ret[:reference_return][param.name.to_sym] = res[indx]
        end
      }
      p = Pathname::new("#{@tmp_dir}/#{@procedure.name}/#{base_name}")
      p.children.each { |f| File::unlink f }
      return boast_ret
    end

    def build_executable(options={})
      compiler_options = BOAST::get_compiler_options
      compiler_options.update(options)
      @probes = [TimerProbe]
      linker, _, _, ldflags = setup_compilers(@probes, compiler_options)
      @compiler_options = compiler_options

      @marker = Tempfile::new([@procedure.name,""])
      @tmp_dir = Dir::mktmpdir

      kernel_files = get_sub_kernels

      create_library_source

      create_executable_source

      save_source

      create_executable_target(linker, ldflags, kernel_files)

      instance_eval <<EOF
      def run(*args, &block)
        run_executable(*args, &block)
      end
EOF

      return self

    end

    public

    def build(options={})
      return build_executable(options) if executable? and (@lang == C or @lang == FORTRAN)
      compiler_options = BOAST::get_compiler_options
      compiler_options.update(options)
      @probes = []
      if compiler_options[:probes] then
        @probes = compiler_options[:probes]
      elsif get_lang != CUDA then
        @probes = [TimerProbe, PAPIProbe]
        @probes.push EnergyProbe if EnergyProbe
        @probes.push AffinityProbe unless OS.mac?
      end
      @probes = [MPPAProbe] if @architecture == MPPA
      linker, ldshared, ldshared_flags, ldflags = setup_compilers(@probes, compiler_options)
      @compiler_options = compiler_options

      @marker = Tempfile::new([@procedure.name,""])

      extend MAQAO if @compiler_options[:MAQAO_PASS]

      kernel_files = get_sub_kernels

      create_sources

      save_source

      create_targets(linker, ldshared, ldshared_flags, ldflags, kernel_files)

      save_binary

      save_module

      load_module

      cleanup(kernel_files) unless keep_temp

      eval "self.extend(#{module_name})"

      return self
    end

    def dump_binary(path = nil)
      f = path ? File::open(path,"wb") : File::open(library_object,"wb")
      @binary.rewind
      f.write( @binary.read )
      f.close
    end

    def dump_source(path = nil)
      f = path ? File::open(path,"wb") : File::open(library_source,"wb")
      @source.rewind
      f.write( @source.read )
      f.close
    end

    def dump_module(path = nil)
      f = path ? File::open(path,"wb") : File::open(module_file_path,"wb")
      @module_binary.rewind
      f.write( @module_binary.read )
      f.close
    end

    def reload_module
      raise "Missing binary library data!" unless @module_binary
      $LOADED_FEATURES.delete(module_file_path)
      require module_file_path
    end

  end

end
