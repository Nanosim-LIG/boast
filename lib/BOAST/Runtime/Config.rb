require 'yaml'

module BOAST

  @@compiler_default_options = {
    :FC => 'gfortran',
    :FCFLAGS => '-O2 -Wall',
    :CC => 'gcc',
    :CFLAGS => '-O2 -Wall',
    :CXX => 'g++',
    :CXXFLAGS => '-O2 -Wall',
    :NVCC => 'nvcc',
    :NVCCFLAGS => '-O2',
    :LDFLAGS => '',
    :CLFLAGS => '',
    :CLVENDOR => nil,
    :CLPLATFORM => nil,
    :CLDEVICE => nil,
    :CLDEVICETYPE => nil,
    :MAQAO => 'maqao',
    :MAQAO_FLAGS => '',
    :probes => nil,
    :openmp => false
  }
  
  @@openmp_default_flags = {
    "gcc" => "-fopenmp",
    "icc" => "-openmp",
    "gfortran" => "-fopenmp",
    "ifort" => "-openmp",
    "g++" => "-fopenmp",
    "icpc" => "-openmp"
  }

  @@run_options = [
    :PAPI
  ]

  @@run_config = {
  }

  module PrivateStateAccessor
    private_boolean_state_accessor :verbose
    private_boolean_state_accessor :debug_source
    private_boolean_state_accessor :ffi
    private_boolean_state_accessor :keep_temp
    private_state_accessor :fortran_line_length
  end

  boolean_state_accessor :verbose
  boolean_state_accessor :debug_source
  boolean_state_accessor :ffi
  boolean_state_accessor :keep_temp
  state_accessor         :fortran_line_length
  default_state_getter :verbose,             false
  default_state_getter :debug_source,        false
  default_state_getter :ffi,                 false
  default_state_getter :keep_temp,           false
  default_state_getter :fortran_line_length, 72

  module_function

  def read_boast_compiler_config
    boast_config_dir = assert_boast_config_dir
    return unless boast_config_dir
    compiler_options_file = "#{boast_config_dir}/compiler_options"
    if File::exist?( compiler_options_file ) then
      File::open( compiler_options_file, "r" ) { |f|
        @@compiler_default_options.update( YAML::load( f.read ) )
      }
    else
      File::open( compiler_options_file, "w" ) { |f|
        f.write YAML::dump( @@compiler_default_options )
      }
    end
    openmp_flags_file = "#{boast_config_dir}/openmp_flags"
    if File::exist?( openmp_flags_file ) then
      File::open( openmp_flags_file, "r" ) { |f|
        @@openmp_default_flags.update( YAML::load( f.read ) )
      }
    else
      File::open( openmp_flags_file, "w" ) { |f|
        f.write YAML::dump( @@openmp_default_flags )
      }
    end
    @@compiler_default_options.each_key { |k|
      @@compiler_default_options[k] = ENV[k.to_s] if ENV[k.to_s]
    }
    @@compiler_default_options[:LD] = ENV["LD"] if ENV["LD"]
  end

  read_boast_compiler_config

  def get_openmp_flags
    return @@openmp_default_flags.clone
  end

  def get_compiler_options
    return @@compiler_default_options.clone
  end

  def read_boast_run_config
    boast_config_dir = assert_boast_config_dir
    run_config_file = "#{boast_config_dir}/run_config"
    if File::exist?( run_config_file ) then
      File::open( run_config_file, "r" ) { |f|
        @@run_config.update( YAML::load( f.read ) )
      }
    else
      File::open( run_config_file, "w" ) { |f|
        f.write YAML::dump( @@run_config )
      }
    end
    @@run_options.each { |o|
      @@run_config[o] = YAML::load(ENV[o.to_s]) if ENV[o.to_s]
    }
  end

  read_boast_run_config

  def get_run_config
    return @@run_config.clone
  end

end
