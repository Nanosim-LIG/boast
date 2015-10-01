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

  module PrivateStateAccessor
    private_boolean_state_accessor :verbose
    private_boolean_state_accessor :debug_source
    private_boolean_state_accessor :ffi
  end

  boolean_state_accessor :verbose
  boolean_state_accessor :debug_source
  boolean_state_accessor :ffi
  @@ffi = false
  @@verbose = false
  @@debug_source = false
  FORTRAN_LINE_LENGTH = 72

  module_function

  def read_boast_config
    home_config_dir = ENV["XDG_CONFIG_HOME"]
    home_config_dir = "#{Dir.home}/.config" if not home_config_dir
    Dir.mkdir( home_config_dir ) if not File::exist?( home_config_dir )
    return if not File::directory?(home_config_dir)
    boast_config_dir = "#{home_config_dir}/BOAST"
    Dir.mkdir( boast_config_dir ) if not File::exist?( boast_config_dir )
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
    @@verbose = ENV["VERBOSE"] if ENV["VERBOSE"]
    @@ffi = ENV["FFI"] if ENV["FFI"]
    @@debug_source = ENV["DEBUG_SOURCE"] if ENV["DEBUG_SOURCE"]
  end

  read_boast_config

  def get_openmp_flags
    return @@openmp_default_flags.clone
  end

  def get_compiler_options
    return @@compiler_default_options.clone
  end

end
