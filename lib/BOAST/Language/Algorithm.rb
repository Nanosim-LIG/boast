module BOAST

  extend TypeTransition
  
  module PrivateStateAccessor

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

    def get_lang_name
      BOAST::get_lang_name
    end

    def annotate_number(*args)
      BOAST::annotate_number(*args)
    end

  end

  module_function

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

  def get_lang_name
    case lang
    when C
      return :C
    when FORTRAN
      return :FORTRAN
    when CL
      return :CL
    when CUDA
      return :CUDA
    else
      nil
    end
  end

  @@output = STDOUT
  @@chain_code = false
  @@decl_module = false
  @@annotate_numbers = Hash::new { |h,k| h[k] = 0 }

  @@env = Hash::new{|h, k| h[k] = []}

  def push_env(vars = {}, &block)
    keys = []
    vars.each { |key, value|
      var = nil
      begin
        var = BOAST::class_variable_get("@@"+key.to_s)
      rescue
        BOAST::pop_env(*keys)
        raise "Unknown module variable #{key}!"
      end
      @@env[key].push(var)
      BOAST::class_variable_set("@@"+key.to_s, value)
      keys.push(key)
    }
    if block then
      begin
        block.call
      ensure
        BOAST::pop_env(*vars.keys)
      end
    end
  end

  def pop_env(*vars)
    vars.each { |key|
      raise "Unknown module variable #{key}!" unless @@env.has_key?(key)
      ret = @@env[key].pop
      raise "No stored value for #{key}!" if ret.nil?
      BOAST::class_variable_set("@@"+key.to_s, ret)
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

  def pr(a, *args)
    pr_annotate(a) if annotate?
    a.pr(*args)
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
       v = BOAST::Variable::new("#{self}", BOAST::Int, :signed => true, :constant => self )
     else
       v = BOAST::Variable::new("#{self}", BOAST::Int, :signed => false, :constant => self )
    end
    v.force_replace_constant = true
    return v
  end
end

class Float
  def to_var
    v = BOAST::Variable::new("#{self}", BOAST::Real, :constant => self )
    v.force_replace_constant = true
    return v
  end
end

