module BOAST

  extend TypeTransition
  
  FORTRAN = 1
  C = 2
  CL = 3
#  OpenCL = 3
  CUDA = 4
  X86 = 1
  ARM = 2

  def BOAST::get_default_lang
    lang = BOAST::const_get(ENV["BOAST_LANG"]) if ENV["BOAST_LANG"]
    return lang if lang
    return BOAST::FORTRAN
  end

  def BOAST::get_default_debug
    debug = false
    debug = ENV["DEBUG"] if ENV["DEBUG"]
    return debug
  end

  @@output = STDOUT
  @@lang = BOAST::get_default_lang
  @@replace_constants = true
  @@default_int_size = 4
  @@default_int_signed = true
  @@default_real_size = 8
  @@indent_level = 0
  @@indent_increment = 2
  @@array_start = 1
  @@chain_code = false
  @@architecture = X86
  @@debug = BOAST::get_default_debug

  @@env = Hash.new{|h, k| h[k] = []}

  def BOAST::push_env(vars = {})
    vars.each { |key,value|
      var = nil
      begin
        var = BOAST::class_variable_get("@@"+key.to_s)
      rescue
        raise "Unknown module variable #{key}!"
      end
      @@env[key].push(var)
      BOAST::class_variable_set("@@"+key.to_s, value)
    }
  end

  def BOAST::pop_env(*vars)
    vars.each { |key|
      raise "Unknown module variable #{key}!" unless @@env.has_key?(key)
      ret = @@env[key].pop
      raise "No stored value for #{key}!" if ret.nil?
      BOAST::class_variable_set("@@"+key.to_s, ret)
    }
  end

  def BOAST::print(a)
    a.print
  end

  def BOAST::decl(*a)
    a.each { |d|
      d.decl
    }
  end

  def BOAST::close(a)
    a.close
  end

  def BOAST::debug=(debug)
    @@debug = debug
  end

  def BOAST::debug
    return @@debug
  end

  def BOAST::architecture=(arch)
    @@architecture = arch
  end

  def BOAST::set_architecture(arch)
    @@architecture = arch
  end

  def BOAST::architecture
    return @@architecture
  end

  def BOAST::get_architecture
    return @@architecture
  end

  def BOAST::indent_level=(level)
    @@indent_level = level
  end

  def BOAST::set_indent_level(level)
    @@indent_level = level
  end

  def BOAST::indent_level
    return @@indent_level
  end

  def BOAST::get_indent_level
    return @@indent_level
  end

  def BOAST::indent_increment
    return @@indent_increment
  end

  def BOAST::get_indent_increment
    return @@indent_increment
  end

  def BOAST::increment_indent_level(increment = @@indent_increment)
    @@indent_level += increment
  end
  
  def BOAST::decrement_indent_level(increment = @@indent_increment)
    @@indent_level -= increment
  end
  
  def BOAST::set_replace_constants(replace_constants)
    @@replace_constants = replace_constants
  end

  def BOAST::replace_constants?
    return @@replace_constants
  end

  def BOAST::get_replace_constants
    return @@replace_constants
  end

  def BOAST::default_int_signed=(signed)
    @@default_int_signed = signed
  end

  def BOAST::set_default_int_signed(signed)
    @@default_int_signed = signed
  end

  def BOAST::default_int_signed?
    return @@default_int_signed
  end

  def BOAST::get_default_int_signed
    return @@default_int_signed
  end

  def BOAST::default_int_size=(size)
    @@default_int_size = size
  end

  def BOAST::set_default_int_size(size)
    @@default_int_size = size
  end

  def BOAST::default_int_size
    return @@default_int_size
  end

  def BOAST::get_default_int_size
    return @@default_int_size
  end

  def BOAST::default_real_size=(size)
    @@default_real_size = size
  end

  def BOAST::set_default_real_size(size)
    @@default_real_size = size
  end

  def BOAST::default_real_size
    return @@default_real_size
  end

  def BOAST::get_default_real_size
    return @@default_real_size
  end

  def BOAST::lang=(lang)
    @@lang = lang
  end

  def BOAST::set_lang(lang)
    @@lang = lang
  end

  def BOAST::lang
    return @@lang
  end

  def BOAST::get_lang
    return @@lang
  end

  def BOAST::output(output)
    @@output = output
  end

  def BOAST::set_output(output)
    @@output = output
  end

  def BOAST::output
    return @@output
  end

  def BOAST::get_output
    return @@output
  end

  def BOAST::set_chain_code(chain_code)
    @@chain_code = chain_code
  end

  def BOAST::get_chain_code
    return @@chain_code
  end

  def BOAST::array_start=(array_start)
    @@array_start = array_start
  end

  def BOAST::set_array_start(array_start)
    @@array_start = array_start
  end

  def BOAST::array_start
    return @@array_start
  end

  def BOAST::get_array_start
    return @@array_start
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

