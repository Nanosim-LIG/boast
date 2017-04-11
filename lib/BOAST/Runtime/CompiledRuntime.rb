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
    attr_reader :param_struct

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

    def save_executable
      f = File::open(target_executable, "rb")
      @executable = StringIO::new
      @executable.write( f.read )
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
      get_output.puts "#include <pthread.h>" unless executable?
      get_output.puts "#include <sys/types.h>" unless executable?
      get_output.puts "#include \"ruby/thread.h\"" unless executable?
      @includes.each { |inc|
        get_output.puts "#include \"#{inc}\""
      }
    end

    def fill_module_preamble
      get_output.print <<EOF
VALUE #{module_name} = Qnil;

static VALUE method_run(int _boast_argc, VALUE *_boast_argv, VALUE _boast_self);

void Init_#{module_name}();
void Init_#{module_name}() {
  #{module_name} = rb_define_module("#{module_name}");
  rb_define_method(#{module_name}, "__run", method_run, -1);
}

static VALUE _boast_check_get_options(int _boast_argc, VALUE *_boast_argv);
static VALUE _boast_check_get_options(int _boast_argc, VALUE *_boast_argv) {
  VALUE _boast_rb_opts = Qnil;
  _boast_rb_opts = _boast_argv[_boast_argc -1];
  return _boast_rb_opts;
}

static int _boast_get_repeat(VALUE _boast_rb_opts);
static int _boast_get_repeat(VALUE _boast_rb_opts) {
  int _boast_repeat = 1;
  if ( _boast_rb_opts != Qnil ){
    VALUE _boast_repeat_value = Qnil;
    _boast_repeat_value = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("repeat")));
    if(_boast_repeat_value != Qnil)
      _boast_repeat = NUM2UINT(_boast_repeat_value);
    if(_boast_repeat < 0)
      _boast_repeat = 1;
  }
  return _boast_repeat;
}
EOF

    if !executable? then
        get_output.print <<EOF
struct _boast_synchro_struct {
  volatile int * counter;
EOF
        if get_synchro == 'MUTEX' then
          get_output.print <<EOF
  pthread_mutex_t * mutex;
  pthread_cond_t * condition;
EOF
        else
          get_output.print <<EOF
  pthread_spinlock_t * spin;
EOF
        end
        get_output.print <<EOF
};
static int _boast_get_coexecute(VALUE _boast_rb_opts, struct _boast_synchro_struct * _boast_synchro);
static int _boast_get_coexecute(VALUE _boast_rb_opts, struct _boast_synchro_struct * _boast_synchro) {
  int _boast_coexecute = 0;
  if ( _boast_rb_opts != Qnil ){
    VALUE _boast_coexecute_value = Qnil;
    _boast_coexecute_value = rb_hash_aref(_boast_rb_opts, ID2SYM(rb_intern("coexecute")));
    if(_boast_coexecute_value != Qnil) {
      VALUE address;
      _boast_coexecute = 1;
      address = rb_funcall(rb_ary_entry(_boast_coexecute_value, 0), rb_intern("address"), 0);
      _boast_synchro->counter = sizeof(_boast_synchro->counter) == 4 ? (void *) NUM2ULONG(address) : (void *) NUM2ULL(address);
EOF
      if get_synchro == 'MUTEX' then
        get_output.print <<EOF
      address = rb_funcall(rb_ary_entry(_boast_coexecute_value, 1), rb_intern("address"), 0);
      _boast_synchro->mutex = sizeof(_boast_synchro->mutex) == 4 ? (void *) NUM2ULONG(address) : (void *) NUM2ULL(address);
      address = rb_funcall(rb_ary_entry(_boast_coexecute_value, 2), rb_intern("address"), 0);
      _boast_synchro->condition = sizeof(_boast_synchro->condition) == 4 ? (void *) NUM2ULONG(address) : (void *) NUM2ULL(address);
EOF
      else
        get_output.print <<EOF
      address = rb_funcall(rb_ary_entry(_boast_coexecute_value, 1), rb_intern("address"), 0);
      _boast_synchro->spin = sizeof(_boast_synchro->spin) == 4 ? (void *) NUM2ULONG(address) : (void *) NUM2ULL(address);
EOF
      end
      get_output.print <<EOF
    }
  }
  return _boast_coexecute;
}

EOF
      end
    end

    def fill_check_args
      get_output.print <<EOF
  VALUE _boast_rb_opts = Qnil;
  _boast_rb_opts = _boast_check_get_options( _boast_argc, _boast_argv);
EOF
    end

    def add_run_options
      get_output.print <<EOF
  _boast_params._boast_repeat = _boast_get_repeat( _boast_rb_opts );
EOF
      get_output.puts "  _boast_params._boast_coexecute = _boast_get_coexecute( _boast_rb_opts, &_boast_params._boast_synchro );" unless executable?
    end

    def fill_param_struct
      pars = @procedure.parameters.collect { |param|
        param.copy(param.name, :const => nil, :constant => nil, :dir => nil, :direction => nil, :reference => nil )
      }
      pars.push @procedure.properties[:return].copy("_boast_ret") if @procedure.properties[:return]
      pars.push Int("_boast_repeat")
      pars.push Int("_boast_coexecute")
      pars.push CStruct("_boast_timer", :type_name => "_boast_timer_struct", :members => [Int(:dummy)]) if @probes.include?(TimerProbe)
      pars.push CStruct("_boast_synchro", :type_name => "_boast_synchro_struct", :members => [Int(:dummy)]) unless executable?
      @param_struct = CStruct("_boast_params", :type_name => "_boast_#{@procedure.name}_params", :members => pars)
    end

    def fill_decl_module_params
      push_env(:decl_module => true) {
        param_struct.decl
        get_output.puts "  VALUE _boast_stats = rb_hash_new();"
        get_output.puts "  VALUE _boast_rb_ptr = Qnil;"
        refs = false
        @procedure.parameters.each { |param|
          refs = true if param.scalar_output?
        }
        if refs then
          get_output.puts "  VALUE _boast_refs = rb_hash_new();"
          get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"reference_return\")),_boast_refs);"
        end
      }
    end

    def copy_scalar_param_from_ruby(str_par, param, ruby_param )
      case param.type
      when Int
        (str_par === FuncCall::new("NUM2INT", ruby_param)).pr if param.type.size == 4
        (str_par === FuncCall::new("NUM2LONG", ruby_param)).pr if param.type.size == 8
      when Real
        (str_par === FuncCall::new("NUM2DBL", ruby_param)).pr
      end
    end

    def copy_array_param_from_ruby(str_par, param, ruby_param )
      rb_ptr = Variable::new("_boast_rb_ptr", CustomType, :type_name => "VALUE")
      (rb_ptr === ruby_param).pr
      get_output.print <<EOF
  if (TYPE(_boast_rb_ptr) == T_STRING) {
    #{
  if param.dimension then
    "#{str_par} = (void *)RSTRING_PTR(_boast_rb_ptr)"
  else
    (str_par === param.copy("*(void *)RSTRING_PTR(_boast_rb_ptr)", :dimension => Dim(), :vector_length => 1)).to_s
  end
    };
  } else if ( IsNArray(_boast_rb_ptr) ) {
    struct NARRAY *_boast_n_ary;
    Data_Get_Struct(_boast_rb_ptr, struct NARRAY, _boast_n_ary);
    #{
  if param.dimension then
    "#{str_par} = (void *) _boast_n_ary->ptr"
  else
    (str_par === param.copy("*(void *) _boast_n_ary->ptr", :dimension => Dim(), :vector_length => 1)).to_s
  end
    };
  } else {
    rb_raise(rb_eArgError, "Wrong type of argument for #{param}, expecting array!");
  }
EOF
    end

    def get_params_value
      argc = @procedure.parameters.length
      argv = Variable::new("_boast_argv", CustomType, :type_name => "VALUE", :dimension => [ Dimension::new(0,argc-1) ] )
      push_env(:decl_module => true) {
        @procedure.parameters.each_index do |i|
          param = @procedure.parameters[i]
          par = param_struct.struct_reference(param_struct.type.members[param.name.to_s])
          if not param.dimension? and not param.vector? then
            copy_scalar_param_from_ruby(par, param, argv[i])
          else
            copy_array_param_from_ruby(par, param, argv[i])
          end
        end
      }
    end

    def create_wrapper
      get_output.print <<EOF
static void * boast_wrapper( void * data ) {
  struct _boast_#{@procedure.name}_params * _boast_params;
  _boast_params = data;
EOF
      if get_synchro == 'MUTEX' then
        get_output.print <<EOF
  pthread_mutex_lock(_boast_params->_boast_synchro.mutex);
  *_boast_params->_boast_synchro.counter -= 1;
  if( *_boast_params->_boast_synchro.counter == 0 ) {
    pthread_cond_broadcast( _boast_params->_boast_synchro.condition );
  }
  while( *_boast_params->_boast_synchro.counter ) {
    pthread_cond_wait( _boast_params->_boast_synchro.condition, _boast_params->_boast_synchro.mutex );
  }
  pthread_mutex_unlock(_boast_params->_boast_synchro.mutex);
EOF
      else
        get_output.print <<EOF
  pthread_spin_lock(_boast_params->_boast_synchro.spin);
  *_boast_params->_boast_synchro.counter -= 1;
  while( *_boast_params->_boast_synchro.counter ) {
    pthread_spin_unlock(_boast_params->_boast_synchro.spin);
    pthread_spin_lock(_boast_params->_boast_synchro.spin);
  }
  pthread_spin_unlock(_boast_params->_boast_synchro.spin);
EOF
      end
      get_output.puts "  _boast_timer_start(&_boast_params->_boast_timer);" if @probes.include?(TimerProbe)
      create_procedure_indirect_call
      get_output.puts "  _boast_timer_stop(&_boast_params->_boast_timer);" if @probes.include?(TimerProbe)
      get_output.print <<EOF
  return NULL;
}
EOF
    end

    def create_procedure_wrapper_call
      get_output.print <<EOF
    rb_thread_call_without_gvl(boast_wrapper, &_boast_params, RUBY_UBF_PROCESS, NULL);
EOF
    end

    def create_procedure_indirect_call
      get_output.puts  "  int _boast_i;"
      get_output.puts  "  for(_boast_i = 0; _boast_i < _boast_params->_boast_repeat; ++_boast_i){"
      get_output.print "    "
      get_output.print "_boast_params->_boast_ret = " if @procedure.properties[:return]
      get_output.print "#{method_name}( "
      get_output.print create_procedure_indirect_call_parameters.join(", ")
      get_output.puts " );"
      get_output.puts  "  }"
    end

    def create_procedure_call
      If("_boast_params._boast_coexecute" => lambda {
        create_procedure_wrapper_call
      }, :else => lambda {
        TimerProbe.start if @probes.include?(TimerProbe)
        get_output.puts  "    int _boast_i;"
        get_output.puts  "    for(_boast_i = 0; _boast_i < _boast_params._boast_repeat; ++_boast_i){"
        get_output.print "      "
        get_output.print "_boast_params._boast_ret = " if @procedure.properties[:return]
        get_output.print "#{method_name}( "
        get_output.print create_procedure_call_parameters.join(", ")
        get_output.puts " );"
        get_output.puts  "    }"
        TimerProbe.stop if @probes.include?(TimerProbe)
      }).pr
    end

    def copy_scalar_param_to_ruby(str_par, param, ruby_param)
      if param.scalar_output? then
        case param.type
        when Int
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_int_new((long long)#{str_par}));" if param.type.signed?
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_int_new((unsigned long long)#{str_par}));" if not param.type.signed?
        when Real
          get_output.puts "  rb_hash_aset(_boast_refs, ID2SYM(rb_intern(\"#{param}\")),rb_float_new((double)#{str_par}));"
        end
      end
    end

    def copy_scalar_param_to_file(str_par, param, base_path)
      if param.scalar_output? then
        get_output.puts <<EOF
  __boast_f = fopen("#{base_path}/#{param}.out", "wb");
  fwrite(&(#{str_par}), sizeof(#{str_par}), 1, __boast_f);
  fclose(__boast_f);
EOF
      end
    end

    def copy_scalar_param_from_file(str_par, param, base_path)
      get_output.puts <<EOF
  __boast_f = fopen("#{base_path}/#{param}.in", "rb");
  if( fread(&(#{str_par}), sizeof(#{str_par}), 1, __boast_f) != 1 ) {
    exit(-1);
  }
  fclose(__boast_f);
EOF
    end

    def copy_array_param_to_ruby(str_par, param, ruby_param)
    end

    def copy_array_param_to_file(str_par, param, base_path)
      if param.direction == :out or param.direction == :inout then
        get_output.puts <<EOF
  __boast_f = fopen("#{base_path}/#{param}.out", "wb");
  fwrite(#{str_par}, 1, __boast_sizeof_#{param}, __boast_f);
  fclose(__boast_f);
  free(#{str_par});
EOF
      else
        get_output.puts <<EOF
  free(#{str_par});
EOF
      end
    end

    def copy_array_param_from_file(str_par, param, base_path)
      get_output.puts <<EOF
  __boast_f = fopen("#{base_path}/#{param}.in", "rb");
  fseek(__boast_f, 0L, SEEK_END);
  __boast_sizeof_#{param} = ftell(__boast_f);
  rewind(__boast_f);
  #{str_par} = malloc(__boast_sizeof_#{param});
  if( fread(#{str_par}, 1, __boast_sizeof_#{param}, __boast_f) != __boast_sizeof_#{param} ) {
    exit(-1);
  }
  fclose(__boast_f);
EOF
    end

    def get_results
      argc = @procedure.parameters.length
      argv = Variable::new("_boast_argv", CustomType, :type_name => "VALUE", :dimension => [ Dimension::new(0,argc-1) ] )
      push_env(:decl_module => true) {
        @procedure.parameters.each_index do |i|
          param = @procedure.parameters[i]
          par = param_struct.struct_reference(param_struct.type.members[param.name.to_s])
          if not param.dimension then
            copy_scalar_param_to_ruby(par, param, argv[i])
          else
            copy_array_param_to_ruby(par, param, argv[i])
          end
        end
      }
    end

    def store_results
      if @procedure.properties[:return] then
        type_ret = @procedure.properties[:return].type
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_int_new((long long)_boast_params._boast_ret));" if type_ret.kind_of?(Int) and type_ret.signed
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_int_new((unsigned long long)_boast_params._boast_ret));" if type_ret.kind_of?(Int) and not type_ret.signed
        get_output.puts "  rb_hash_aset(_boast_stats,ID2SYM(rb_intern(\"return\")),rb_float_new((double)_boast_params._boast_ret));" if type_ret.kind_of?(Real)
      end
    end

    def get_executable_params_value( base_path )
      push_env(:decl_module => true) {
        @procedure.parameters.each do |param|
          par = param_struct.struct_reference(param_struct.type.members[param.name.to_s])
          if not param.dimension? then
            copy_scalar_param_from_file(par, param, base_path)
          else
            copy_array_param_from_file(par, param, base_path)
          end
        end
      }
    end

    def get_executable_params_return_value( base_path )
      push_env(:decl_module => true) {
        @procedure.parameters.each do |param|
          par = param_struct.struct_reference(param_struct.type.members[param.name.to_s])
          if not param.dimension then
            copy_scalar_param_to_file(par, param, base_path)
          else
            copy_array_param_to_file(par, param, base_path)
          end
        end
      }
    end

    def fill_executable_source
      fill_param_struct
      get_output.puts "#include <inttypes.h>"
      get_output.puts "#include <stdlib.h>"
      get_output.puts "#include <stdio.h>"
      @includes.each { |inc|
        get_output.puts "#include \"#{inc}\""
      }
      @probes.map(&:header)
      @procedure.boast_header(@lang)
      @probes.map(&:preamble)

      param_struct.type.define
      get_output.print <<EOF
void Init_#{base_name}( int _boast_repeat );
void Init_#{base_name}( int _boast_repeat ) {
EOF
      increment_indent_level
      output.puts "  FILE * __boast_f;"
      param_struct.decl
      push_env(:decl_module => true) {
        @procedure.parameters.each { |param|
          if param.dimension? then
            output.puts "  size_t __boast_sizeof_#{param};"
          end
        }
      }
      @probes.reject{ |e| e ==TimerProbe }.reverse.map(&:decl)
      @probes.map(&:configure)

      get_executable_params_value( "#{@tmp_dir}/#{@procedure.name}/#{base_name}" )

      @probes.reverse.map(&:start)

      get_output.puts  "  int _boast_i;"
      get_output.puts  "  for(_boast_i = 0; _boast_i < _boast_repeat; ++_boast_i){"
      get_output.print "    "
      get_output.print "_boast_params._boast_ret = " if @procedure.properties[:return]
      get_output.print "#{method_name}( "
      get_output.print create_procedure_call_parameters.join(", ")
      get_output.puts  " );"
      get_output.puts  "  }"

      @probes.map(&:stop)

      get_output.puts '  printf("---\n");'
      if @procedure.properties[:return] then
        type_ret = @procedure.properties[:return].type
        get_output.puts '  printf(":return: %lld\n", (long long)_boast_params._boast_ret);' if type_ret.kind_of?(Int) and type_ret.signed
        get_output.puts '  printf(":return: %ulld\n", (unsigned long long)_boast_params._boast_ret);' if type_ret.kind_of?(Int) and not type_ret.signed
        get_output.puts '  printf(":return: %lf\n", (double)_boast_params._boast_ret);' if type_ret.kind_of?(Real)

      end

      get_executable_params_return_value( "#{@tmp_dir}/#{@procedure.name}/#{base_name}" )

      @probes.map(&:compute)

      @probes.map(&:to_yaml)

      decrement_indent_level
      get_output.print <<EOF
}
int main(int argc, char * argv[]) {
  Init_#{base_name}(atoi(argv[1]));
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
      fill_param_struct
      fill_module_header
      @probes.map(&:header)
      @procedure.boast_header(@lang)

      fill_module_preamble
      @probes.map(&:preamble)

      set_transition("VALUE", "VALUE", :default,  CustomType::new(:type_name => "VALUE"))

      param_struct.type.define

      create_wrapper

      get_output.puts "static VALUE method_run(int _boast_argc, VALUE *_boast_argv, VALUE _boast_self) {"

      increment_indent_level

      fill_decl_module_params

      fill_check_args

      add_run_options

      @probes.reject{ |e| e == TimerProbe }.reverse.map(&:decl)

      get_params_value

      @probes.map(&:configure)

      @probes.reject{ |e| e == TimerProbe }.reverse.map(&:start)

      create_procedure_call

      @probes.reject{ |e| e == TimerProbe }.map(&:stop)

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

    def target_executable_sources
      return [ executable_source, library_source ]
    end

    def cleanup(kernel_files)
      ([target] + target_depends + target_sources).each { |fn|
        File::unlink(fn)
      }
      kernel_files.each { |f|
        f.unlink
      }
    end

    def cleanup_executable(kernel_files)
      ( [target_executable] + target_executable_depends + target_executable_sources ).each { |fn|
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

      Dir::mkdir(@tmp_dir, 0700) unless Dir::exist?(@tmp_dir)

      dump_executable
      dump_ref_inputs( { base_name => ps }, @tmp_dir )
      boast_ret = YAML::load `#{target_executable} #{options[:repeat]}`
      File::unlink(target_executable) unless keep_temp

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
      unless keep_temp then
        Dir::rmdir("#{@tmp_dir}/#{@procedure.name}/#{base_name}")
        Dir::rmdir("#{@tmp_dir}/#{@procedure.name}")
        Dir::rmdir("#{@tmp_dir}")
      end
      return boast_ret
    end

    def build_executable(options={})
      compiler_options = BOAST::get_compiler_options
      compiler_options.update(options)
      @probes = [TimerProbe]
      linker, _, _, ldflags = setup_compilers(@probes, compiler_options)
      @compiler_options = compiler_options

      @marker = Tempfile::new([@procedure.name,""])
      Dir::mktmpdir { |dir|
        @tmp_dir = "#{dir}"
      }

      kernel_files = get_sub_kernels

      create_library_source

      create_executable_source

      save_source

      create_executable_target(linker, ldflags, kernel_files)

      save_executable

      instance_eval <<EOF
      def run(*args, &block)
        run_executable(*args, &block)
      end
EOF

      cleanup_executable(kernel_files) unless keep_temp

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
        @probes.push AffinityProbe if AffinityProbe
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

      define_singleton_method(:run) { |*args, **options, &block|
        raise "Wrong number of arguments for #{@procedure.name} (#{args.length} for #{@procedure.parameters.length})" if args.length != @procedure.parameters.length
        config = BOAST::get_run_config
        config.update(options)
        res = nil
        if AffinityProbe == HwlocProbe and config[:cpu_affinity] then
          affinity = config[:cpu_affinity]
          if affinity.kind_of?(Array) then
            cpuset = Hwloc::Cpuset::new( affinity )
          elsif affinity.kind_of?(Hwloc::Bitmap) then
            cpuset = affinity
          end
          AffinityProbe.topology.set_cpubind(cpuset, Hwloc::CPUBIND_THREAD | Hwloc::CPUBIND_STRICT) {
            res = __run(*args, config, &block)
          }
        else
          res = __run(*args, config, &block)
        end
        res
      }

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

    def dump_executable(path = nil)
      f = path ? File::open(path,"wb",0700) : File::open(target_executable,"wb",0700)
      @executable.rewind
      f.write( @executable.read )
      f.close
    end

    def reload_module
      raise "Missing binary library data!" unless @module_binary
      $LOADED_FEATURES.delete(module_file_path)
      require module_file_path
    end

  end

end
