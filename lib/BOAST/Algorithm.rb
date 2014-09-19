module BOAST

  extend TypeTransition
  
  FORTRAN = 1
  C = 2
  CL = 3
  CUDA = 4
  X86 = 1
  ARM = 2

  module PrivateStateAccessor

    private_state_accessor :output, :lang, :architecture
    private_state_accessor :default_int_size, :default_real_size
    private_state_accessor :array_start
    private_state_accessor :indent_level, :indent_increment

    private_boolean_state_accessor :replace_constants
    private_boolean_state_accessor :default_int_signed
    private_boolean_state_accessor :chain_code
    private_boolean_state_accessor :debug

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

  end

  state_accessor :output, :lang, :architecture
  state_accessor :default_int_size, :default_real_size
  state_accessor :array_start
  state_accessor :indent_level, :indent_increment

  boolean_state_accessor :replace_constants
  boolean_state_accessor :default_int_signed
  boolean_state_accessor :chain_code
  boolean_state_accessor :debug

  module_function

  def get_default_lang
    lang = const_get(ENV["BOAST_LANG"]) if ENV["BOAST_LANG"]
    return lang if lang
    return FORTRAN
  end

  def get_default_debug
    debug = false
    debug = ENV["DEBUG"] if ENV["DEBUG"]
    return debug
  end

  @@output = STDOUT
  @@lang = get_default_lang
  @@replace_constants = true
  @@default_int_size = 4
  @@default_int_signed = true
  @@default_real_size = 8
  @@indent_level = 0
  @@indent_increment = 2
  @@array_start = 1
  @@chain_code = false
  @@architecture = X86
  @@debug = get_default_debug

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

  def indent
     return " "*get_indent_level
  end

  def pr(a)
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

