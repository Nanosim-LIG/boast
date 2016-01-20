module BOAST

  extend TypeTransition
  
  FORTRAN = 1
  C = 2
  CL = 3
  CUDA = 4
  #X86 = 1
  #ARM = 2
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

    private_boolean_state_accessor :replace_constants
    private_boolean_state_accessor :default_int_signed
    private_boolean_state_accessor :chain_code
    private_boolean_state_accessor :debug
    private_boolean_state_accessor :use_vla
    private_boolean_state_accessor :decl_module
    private_boolean_state_accessor :annotate

    private
    def push_env(*args)
      BOAST::push_env(*args)
    end

    def pop_env(*args)
      BOAST::pop_env(*args)
    end

    def increment_indent_level(*args)
      BOAST::increment_indent_level(*args)
    end

    def decrement_indent_level(*args)
      BOAST::decrement_indent_level(*args)
    end

    def indent
      BOAST::indent
    end

    def get_architecture_name
      BOAST::get_architecture_name
    end

    def annotate_number(*args)
      BOAST::annotate_number(*args)
    end

  end

  state_accessor :output, :lang, :architecture, :model, :address_size
  state_accessor :default_int_size, :default_real_size
  state_accessor :default_align
  state_accessor :array_start
  state_accessor :indent_level, :indent_increment
  state_accessor :annotate_list
  state_accessor :annotate_indepth_list
  state_accessor :annotate_level

  boolean_state_accessor :replace_constants
  boolean_state_accessor :default_int_signed
  boolean_state_accessor :chain_code
  boolean_state_accessor :debug
  boolean_state_accessor :use_vla
  boolean_state_accessor :decl_module
  boolean_state_accessor :annotate

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

  alias use_vla_old? use_vla?
  class << self
    alias use_vla_old? use_vla?
  end

  def use_vla?
    return false if [CL,CUDA].include?(lang)
    return use_vla_old?
  end

  module_function :use_vla?

  module_function

  def get_default_architecture
    architecture = const_get(ENV["ARCHITECTURE"]) if ENV["ARCHITECTURE"]
    architecture = const_get(ENV["ARCH"]) if not architecture and ENV["ARCH"]
    return architecture if architecture
    return ARM if YAML::load( OS.report )["host_cpu"].match("arm")
    return X86
  end

  def get_architecture_name
    case architecture
    when X86
      return :X86
    when ARM
      return :ARM
    when MPPA
      return :MPPA
    else
      return nil
    end
  end

  @@output = STDOUT
  @@chain_code = false
  @@architecture = get_default_architecture
  @@decl_module = false
  @@annotate_numbers = Hash::new { |h,k| h[k] = 0 }

  @@env = Hash::new{|h, k| h[k] = []}

  def push_env(vars = {})
    vars.each { |key,value|
      var = nil
      begin
        var = class_variable_get("@@"+key.to_s)
      rescue
        raise "Unknown module variable #{key}!"
      end
      @@env[key].push(var)
      class_variable_set("@@"+key.to_s, value)
    }
  end

  def pop_env(*vars)
    vars.each { |key|
      raise "Unknown module variable #{key}!" unless @@env.has_key?(key)
      ret = @@env[key].pop
      raise "No stored value for #{key}!" if ret.nil?
      class_variable_set("@@"+key.to_s, ret)
    }
  end

  def increment_indent_level(increment = get_indent_increment)
    set_indent_level( get_indent_level + increment )
  end
  
  def decrement_indent_level(increment = get_indent_increment)
    set_indent_level( get_indent_level - increment )
  end

  def annotate_number(name)
    num = @@annotate_numbers[name]
    @@annotate_numbers[name] = num + 1
    return num
  end

  def indent
     return " "*get_indent_level
  end

  def pr_annotate(a)
    name = a.class.name.gsub("BOAST::","")
    if annotate_list.include?(name) then
      description = nil
      if a.is_a?(Annotation) and a.annotate_indepth?(0) then
        description = a.annotation(0)
      end
      annotation = { "#{name}#{annotate_number(name)}" => description }
      Comment(YAML::dump(annotation)).pr
    end
  end

  def pr(a)
    pr_annotate(a) if annotate?
    a.pr
  end

  def decl(*a)
    a.each { |d|
      d.decl
    }
  end

  def close(a)
    a.close
  end

  def opn(a)
    a.open
  end

  alias :Var :Variable
  alias :Dim :Dimension
  alias :Call :FuncCall

  class << self
    alias :Var :Variable
    alias :Dim :Dimension
    alias :Call :FuncCall
  end

  Var = Variable
  Dim = Dimension
  Call = FuncCall

  set_transition(Int, Int, :default, Int)
  set_transition(Real, Int, :default, Real)
  set_transition(Int, Real, :default, Real)
  set_transition(Real, Real, :default, Real)
  set_transition(Sizet, Sizet, :default, Sizet)
  set_transition(Sizet, Int, :default, Sizet)
  set_transition(Int, Sizet, :default, Sizet)

end

ConvolutionGenerator = BOAST

class Integer
  def to_var
    if self < 0 then
       return BOAST::Variable::new("#{self}", BOAST::Int, :signed => true, :constant => self )
     else
       return BOAST::Variable::new("#{self}", BOAST::Int, :signed => false, :constant => self )
    end
  end
end

class Float
  def to_var
    return BOAST::Variable::new("#{self}", BOAST::Real, :constant => self )
  end
end

