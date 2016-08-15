require 'os'
require 'yaml'

module BOAST
  FORTRAN = 1
  C = 2
  CL = 3
  CUDA = 4
  X86 = 1
  ARM = 2
  MPPA = 3

  @@boast_config = {
    :fortran_line_length => 72
  }

  module_function

  def assert_boast_config_dir
    home_config_dir = ENV["XDG_CONFIG_HOME"]
    home_config_dir = "#{Dir.home}/.config" if not home_config_dir
    Dir.mkdir( home_config_dir ) if not File::exist?( home_config_dir )
    return nil if not File::directory?(home_config_dir)
    boast_config_dir = "#{home_config_dir}/BOAST"
    Dir.mkdir( boast_config_dir ) if not File::exist?( boast_config_dir )
    return nil if not File::directory?(boast_config_dir)
    return boast_config_dir
  end

  def read_boast_config
    boast_config_dir = assert_boast_config_dir
    return unless boast_config_dir
    boast_config_file = "#{boast_config_dir}/config"
    if File::exist?( boast_config_file ) then
      File::open( boast_config_file, "r" ) { |f|
        @@boast_config.update( YAML::load( f.read ) )
      }
    else
      File::open( boast_config_file, "w" ) { |f|
        f.write YAML::dump( @@boast_config )
      }
    end
  end

  read_boast_config

  module PrivateStateAccessor

    private_state_accessor :output
    private_state_accessor :lang
    private_state_accessor :architecture
    private_state_accessor :model
    private_state_accessor :address_size
    private_state_accessor :default_int_size
    private_state_accessor :default_real_size
    private_state_accessor :default_align
    private_state_accessor :array_start
    private_state_accessor :indent_level
    private_state_accessor :indent_increment
    private_state_accessor :annotate_list
    private_state_accessor :annotate_indepth_list
    private_state_accessor :annotate_level
    private_state_accessor :optimizer_log_file

    private_boolean_state_accessor :replace_constants
    private_boolean_state_accessor :default_int_signed
    private_boolean_state_accessor :chain_code
    private_boolean_state_accessor :debug
    private_boolean_state_accessor :use_vla
    private_boolean_state_accessor :decl_module
    private_boolean_state_accessor :annotate
    private_boolean_state_accessor :optimizer_log
    private_boolean_state_accessor :disable_openmp

  end

  state_accessor :output
  state_accessor :lang
  state_accessor :architecture
  state_accessor :model
  state_accessor :address_size
  state_accessor :default_int_size
  state_accessor :default_real_size
  state_accessor :default_align
  state_accessor :array_start
  state_accessor :indent_level
  state_accessor :indent_increment
  state_accessor :annotate_list
  state_accessor :annotate_indepth_list
  state_accessor :annotate_level
  state_accessor :optimizer_log_file

  boolean_state_accessor :replace_constants
  boolean_state_accessor :default_int_signed
  boolean_state_accessor :chain_code
  boolean_state_accessor :debug
  boolean_state_accessor :use_vla
  boolean_state_accessor :decl_module
  boolean_state_accessor :annotate
  boolean_state_accessor :optimizer_log
  boolean_state_accessor :disable_openmp
  boolean_state_accessor :boast_inspect


  default_state_getter :address_size,          OS.bits/8
  default_state_getter :lang,                  FORTRAN, '"const_get(#{envs})"', :BOAST_LANG
  default_state_getter :model,                 "native"
  default_state_getter :debug,                 false
  default_state_getter :use_vla,               false
  default_state_getter :replace_constants,     true
  default_state_getter :default_int_signed,    true
  default_state_getter :default_int_size,      4
  default_state_getter :default_real_size,     8
  default_state_getter :default_align,         1
  default_state_getter :indent_level,          0
  default_state_getter :indent_increment,      2
  default_state_getter :array_start,           1
  default_state_getter :annotate,              false
  default_state_getter :annotate_list,         ["For"], '"#{envs}.split(\",\").collect { |arg| YAML::load(arg) }"'
  default_state_getter :annotate_indepth_list, ["For"], '"#{envs}.split(\",\").collect { |arg| YAML::load(arg) }"'
  default_state_getter :annotate_level,        0
  default_state_getter :optimizer_log,         false
  default_state_getter :optimizer_log_file,    nil
  default_state_getter :disable_openmp,        false
  default_state_getter :boast_inspect, false, nil, :INSPECT

  class << self
    alias use_vla_old? use_vla?
    private :use_vla_old?
  end

  # @return the boolean evaluation of the *use_vla* state. false if lang is CL or CUDA.
  def use_vla?
    return false if [CL,CUDA].include?(lang)
    return use_vla_old?
  end

  def set_model(val)
    @@model=val
    Intrinsics::generate_conversions
  end

  def model=(val)
    @@model=val
    Intrinsics::generate_conversions
  end

  # @private
  def get_default_architecture
    architecture = const_get(ENV["ARCHITECTURE"]) if ENV["ARCHITECTURE"]
    architecture = const_get(ENV["ARCH"]) if not architecture and ENV["ARCH"]
    return architecture if architecture
    return ARM if YAML::load( OS.report )["host_cpu"].match("arm")
    return X86
  end

  @@architecture = get_default_architecture

end
