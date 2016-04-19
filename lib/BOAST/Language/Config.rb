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

  module PrivateStateAccessor

    private_state_accessor :output, :lang, :architecture, :model, :address_size
    private_state_accessor :default_int_size, :default_real_size
    private_state_accessor :default_align
    private_state_accessor :array_start
    private_state_accessor :indent_level, :indent_increment
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

  state_accessor :output, :lang, :architecture, :model, :address_size
  state_accessor :default_int_size, :default_real_size
  state_accessor :default_align
  state_accessor :array_start
  state_accessor :indent_level, :indent_increment
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

  alias use_vla_old? use_vla?
  class << self
    alias use_vla_old? use_vla?
  end

  def use_vla?
    return false if [CL,CUDA].include?(lang)
    return use_vla_old?
  end

  module_function :use_vla?

  def get_default_architecture
    architecture = const_get(ENV["ARCHITECTURE"]) if ENV["ARCHITECTURE"]
    architecture = const_get(ENV["ARCH"]) if not architecture and ENV["ARCH"]
    return architecture if architecture
    return ARM if YAML::load( OS.report )["host_cpu"].match("arm")
    return X86
  end

  module_function :get_default_architecture

  @@architecture = get_default_architecture

end
