module BOAST

  extend TypeTransition

  EXTENDED.push TypeTransition
  
  module PrivateStateAccessor

    private
    # (see BOAST#push_env)
    def push_env(vars, &block)
      BOAST::push_env(vars, &block)
    end

    # (see BOAST#pop_env)
    def pop_env(*vars)
      BOAST::pop_env(*vars)
    end

    # (see BOAST#increment_indent_level)
    def increment_indent_level(increment = get_indent_increment)
      BOAST::increment_indent_level(increment)
    end

    # (see BOAST#decrement_indent_level)
    def decrement_indent_level(increment = get_indent_increment)
      BOAST::decrement_indent_level(increment)
    end

    # (see BOAST#indent)
    def indent
      BOAST::indent
    end

    # (see BOAST#get_architecture_name)
    def get_architecture_name
      BOAST::get_architecture_name
    end

    # (see BOAST#get_lang_name)
    def get_lang_name
      BOAST::get_lang_name
    end

    # (see BOAST#annotate_number)
    def annotate_number(name)
      BOAST::annotate_number(name)
    end

  end

  module_function

  # Returns the symbol corresponding to the active architecture
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

  # Returns the symbol corresponding to the active language
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

  # Updates states and stores their value in a stack for later retrieval
  # @overload push_env( vars )
  # @overload push_env( vars, &block )
  # @param [Hash] vars contains state symbols and values pairs
  # @yield states will be popped after the given block is called
  def push_env(vars, &block)
    keys = []
    vars.each { |key, value|
      var = nil
      begin
        var = BOAST::send("get_#{key}")
      rescue
        BOAST::pop_env(*keys)
        raise "Unknown module variable #{key}!"
      end
      @@env[key].push(var)
      BOAST::send("set_#{key}", value)
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

  # Pops the specified states values
  # @param vars a list of state symbols
  def pop_env(*vars)
    vars.each { |key|
      raise "Unknown module variable #{key}!" unless @@env.has_key?(key)
      ret = @@env[key].pop
      raise "No stored value for #{key}!" if ret.nil?
      BOAST::send("set_#{key}", ret)
    }
  end

  # Increments the indent level
  # @param [Integer] increment number of space to add
  def increment_indent_level(increment = get_indent_increment)
    set_indent_level( get_indent_level + increment )
  end

  # Decrements the indent level
  # @param [Integer] increment number of space to remove 
  def decrement_indent_level(increment = get_indent_increment)
    set_indent_level( get_indent_level - increment )
  end

  # Returns a string with as many space as the indent level.
  def indent
     return " "*get_indent_level
  end

  # Returns an annotation number for the given name. The number
  # is incremented for a given name is incremented each time this name is called
  def annotate_number(name)
    num = @@annotate_numbers[name]
    @@annotate_numbers[name] = num + 1
    return num
  end

  # Annotates an Object by inlining a YAML structure in a comment.
  # If object's class is part of the annotate list an indepth version of the annotation
  # will be generated.
  # @param [Object] a object to annotate
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


  # One of BOAST keywords: prints BOAST objects.
  # Annotates the given object.
  # Calls the given object pr method with the optional arguments.
  # @param a a BOAST Expression, ControlStructure or Procedure
  # @param args an optional list of parameters
  def pr(a, *args, &block)
    pr_annotate(a) if annotate?
    a.pr(*args, &block)
  end

  # One of BOAST keywords: declares BOAST Variables and Procedures.
  # Calls the decl method of each given objects.
  # @param list a list of parameters do declare
  def decl(*list)
    list.each { |d|
      d.decl
    }
  end

  # One of BOAST keywords: opens a BOAST ControlStructure or Procedure.
  # Calls the open method of the given object.
  # @param a the BOAST object to open
  def opn(a)
    a.open
  end

  # One of BOAST keywords: closes a BOAST ControlStructure or Procedure.
  # Calls the close method of the given object.
  # @param a the BOAST object to close
  def close(a)
    a.close
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
  # Creates a constant BOAST Int Variable with a name corresponding to its value.
  # The variable is signed only when negative.
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
  # Creates a constant BOAST Real Variable with a name corresponding to its value.
  def to_var
    v = BOAST::Variable::new("#{self}", BOAST::Real, :constant => self )
    v.force_replace_constant = true
    return v
  end
end

