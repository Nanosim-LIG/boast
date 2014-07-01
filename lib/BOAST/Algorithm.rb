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

  class Pragma
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    attr_reader :name
    attr_reader :options

    def initialize(name, options)
      @name = name
      @options = options
    end

    def to_s
      self.to_str
    end

    def to_str
      s = ""
      if BOAST::get_lang == FORTRAN then
        s += "$!"
      else
        s += "#pragma"
      end
      @options.each{ |opt|
        s += " #{opt}"
      }
      return s
    end

    def print(final = true)
      s=""
      s += self.to_str
      BOAST::get_output.puts s if final
      return s
    end
  end


  class CodeBlock
     def initialize(&block)
       @block = block
     end

     def print(final=true)
      s=""
      s += " "*BOAST::get_indent_level if final
      BOAST::increment_indent_level
      BOAST::get_output.puts s if final
      if @block then
        s += "\n"
        @block.call
      end
      return s
    end 
  end

  class Dimension
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    attr_reader :val1
    attr_reader :val2
    attr_reader :size
    def initialize(v1=nil,v2=nil)
      @size = nil
      @val1 = nil
      @val2 = nil
      if v2.nil? and v1 then
        @val1 = BOAST::get_array_start
        @val2 = v1 + BOAST::get_array_start - 1
        @size = v1
      else
        @val1 = v1
        @val2 = v2
      end
    end
    def to_str
      s = ""
      if @val2 then
        if BOAST::get_lang == FORTRAN then
          s += @val1.to_s
          s += ":"
          s += @val2.to_s
        elsif [C, CL, CUDA].include?( BOAST::get_lang ) then
          s += (@val2 - @val1 + 1).to_s
        end
      elsif @val1.nil? then
        return nil
      else
        s += @val1.to_s
      end
      return s
    end
    def to_s
      self.to_str
    end
  end


  class ConstArray < Array
    def initialize(array,type = nil)
      super(array)
      @type = type::new if type
    end
    def to_s
      self.to_str
    end
    def to_str
      return self.to_str_fortran if BOAST::get_lang == FORTRAN
      return self.to_str_c if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def to_str_fortran
      s = ""
      return s if self.first.nil?
      s += "(/ &\n"
      s += self.first.to_s
      s += "_wp" if @type and @type.size == 8
      self[1..-1].each { |v|
        s += ", &\n"+v.to_s
        s += "_wp" if @type and @type.size == 8
      }
      s += " /)"
    end
    def to_str_c
      s = ""
      return s if self.first.nil?
      s += "{\n"
      s += self.first.to_s 
      self[1..-1].each { |v|
        s += ",\n"+v.to_s
      }
      s += "}"
    end
  end

  class Ternary
    include BOAST::Arithmetic

    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    attr_reader :operand1
    attr_reader :operand2
    attr_reader :operand3
    
    def initialize(x,y,z)
      @operand1 = x
      @operand2 = y
      @operand3 = z
    end

    def to_s
      self.to_str
    end

    def to_str
      raise "Ternary operator unsupported in FORTRAN!" if BOAST::get_lang == FORTRAN
      return self.to_str_c if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def to_str_c
      s = ""
      s += "(#{@operand1} ? #{@operand2} : #{@operand3})"
    end
    def print(final=true)
      s=""
      s += " "*BOAST::get_indent_level if final
      s += self.to_str
      s += ";" if final and [C, CL, CUDA].include?( BOAST::get_lang )
      BOAST::get_output.puts s if final
      return s
    end

  end
 
  class FuncCall
    include BOAST::Arithmetic

    @return_type
    @options
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    attr_reader :func_name
    attr_reader :args
    attr_accessor :prefix

    def initialize(func_name, *args)
      @func_name = func_name
      if args.last.kind_of?(Hash) then
        @options = args.last
        @args = args[0..-2]
      else
        @args = args
      end
      @return_type = @options[:returns] if @options
    end

    def to_var
      if @return_type then
        if @return_type.kind_of?(Variable)
          return @return_type.copy("#{self}")
        else
          return Variable::new("#{self}", @return_type)
        end
      end
      return nil
    end
      
    def to_s
      self.to_str
    end

    def to_str
      return self.to_str_fortran if BOAST::get_lang == FORTRAN
      return self.to_str_c if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def to_str_fortran
      s = ""
      s += @prefix if @prefix
      s += "#{func_name}(#{@args.join(", ")})"
    end
    def to_str_c
      s = ""
      s += @prefix if @prefix
      s += "#{func_name}(#{@args.join(", ")})"
    end
    def print(final=true)
      s=""
      s += " "*BOAST::get_indent_level if final
      s += self.to_str
      s += ";" if final and [C, CL, CUDA].include?( BOAST::get_lang )
      BOAST::get_output.puts s if final
      return s
    end
  end

  class While
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    attr_reader :condition
    def initialize(condition, &block)
      @condition = condition
      @block = block
    end
    def to_s
      self.to_str
    end
    def to_str
      return self.to_str_fortran if BOAST::get_lang == FORTRAN
      return self.to_str_c if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def to_str_fortran
      s = ""
      s += "do while( #{@condition} )"
      return s
    end
    def to_str_c
      s = ""
      s += "while(#{@condition}){"
      return s
    end
    def print(*args)
      final = true
      s=""
      s += " "*BOAST::get_indent_level if final
      s += self.to_str
      BOAST::increment_indent_level      
      BOAST::get_output.puts s if final
      if @block then
        s += "\n"
        @block.call(*args)
        s += self.close
      end
      return s
    end
    def close(final=true)
      return self.close_fortran(final) if BOAST::get_lang == FORTRAN
      return self.close_c(final) if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def close_c(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "}"
      BOAST::get_output.puts s if final
      return s
    end
    def close_fortran(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "end do"
      BOAST::get_output.puts s if final
      return s
    end

  end

  class Else
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    attr_reader :condition
    def initialize(condition=nil, &block)
      @condition = condition
      @block = block
    end
    def to_s
      self.to_str
    end
    def to_str
      return self.to_str_fortran if BOAST::get_lang == FORTRAN
      return self.to_str_c if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def to_str_fortran
      s = ""
      if @condition then
        s += "else if #{@condition} then"
      else
        s += "else"
      end
      return s
    end
    def to_str_c
      s = ""
      if @condition then
        s += "else if(#{@condition}){"
      else
        s += "else {"
      end
      return s
    end
    def print(*args)
      final = true
      s=""
      s += " "*BOAST::get_indent_level if final
      s += self.to_str
      BOAST::increment_indent_level      
      BOAST::get_output.puts s if final
      if @block then
        s += "\n"
        @block.call(*args)
        s += self.close
      end
      return s
    end
    def close(final=true)
      return self.close_fortran(final) if BOAST::get_lang == FORTRAN
      return self.close_c(final) if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def close_c(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "}"
      BOAST::get_output.puts s if final
      return s
    end
    def close_fortran(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "end if"
      BOAST::get_output.puts s if final
      return s
    end

  end

  class Case
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    attr_reader :expression
    attr_reader :constants_list

    def initialize(expression, *control)
      @expression = expression
      @constants_list = []
      @blocks = []
      if control.size < 1 then
        raise "No block given!"
      elsif control.size.even? then
        (0..control.size-1).step(2) { |i|
          @constants_list[i/2] = [control[i]].flatten
          @blocks[i/2] = control[i+1]
        }
      else
        (0..control.size-2).step(2) { |i|
          @constants_list[i/2] = [control[i]].flatten
          @blocks[i/2] = control[i+1]
        }
        @blocks.push(control.last)
      end
    end

    def to_s(*args)
      self.to_str(*args)
    end

    def to_str(constants, first= true)
      return self.to_str_fortran(constants, first) if BOAST::get_lang == FORTRAN
      return self.to_str_c(constants, first) if [C, CL, CUDA].include?( BOAST::get_lang )
    end

    def to_str_fortran(constants, first)
      s = ""
      if first then
        s += " "*BOAST::get_indent_level
        s += "select case (#{@expression})\n"
        BOAST::increment_indent_level
      else
        BOAST::decrement_indent_level
      end
      s += " "*BOAST::get_indent_level
      if constants and constants.size>0 then
        s += "case (#{constants.join(" : ")})"
      else
        s += "case default"
      end
      BOAST::increment_indent_level
      return s
    end

    def to_str_c(constants, first)
      s = ""
      if first then
        s += " "*BOAST::get_indent_level
        s += "switch(#{@expression}){\n"
        BOAST::increment_indent_level
      else
        s += " "*BOAST::get_indent_level + "break;\n"
        BOAST::decrement_indent_level
      end
      s += " "*BOAST::get_indent_level
      if constants and constants.size>0 then
        s += "case #{constants.join(" : case")} :"
      else
        s += "default :"
      end
      BOAST::increment_indent_level
      return s
    end

    def print(*args)
      first = true
      @blocks.each_index { |indx|
        s = self.to_str(@constants_list[indx],first)
        BOAST::get_output.puts s
        @blocks[indx].call(*args)
        first = false
      }
      self.close
      return self
    end
    def close(final=true)
      return self.close_fortran(final) if BOAST::get_lang == FORTRAN
      return self.close_c(final) if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def close_c(final=true)
      s = ""
      s += " "*BOAST::get_indent_level if final
      s += "break;\n"
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "}"
      BOAST::decrement_indent_level      
      BOAST::get_output.puts s if final
      return s
    end
    def close_fortran(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "end select"
      BOAST::decrement_indent_level      
      BOAST::get_output.puts s if final
      return s
    end

  end 
  class If
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    attr_reader :conditions
    def initialize(*conditions, &block)
      @conditions = []
      @blocks = []
      if conditions.size == 0 then
        raise "Illegal if construct!"
      elsif conditions.size == 1 then
        @conditions.push(conditions[0])
        @blocks.push(block)
      elsif conditions.size.even? then
        (0..conditions.size-1).step(2) { |i|
          @conditions[i/2] = conditions[i]
          @blocks[i/2] = conditions[i+1]
        }
      else
        (0..conditions.size-2).step(2) { |i|
          @conditions[i/2] = conditions[i]
          @blocks[i/2] = conditions[i+1]
        }
        @blocks.push(conditions.last)
      end
    end
    def to_s(*args)
      self.to_str(*args)
    end
    def to_str(condition, first= true)
      return self.to_str_fortran(condition, first) if BOAST::get_lang == FORTRAN
      return self.to_str_c(condition, first) if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def to_str_fortran(condition, first)
      s = ""
      if first then
        s += "if ( #{condition} ) then"
      else
        if condition then
          s += "else if ( #{condition} ) then"
        else
          s += "else"
        end
      end
      return s
    end
    def to_str_c(condition, first)
      s = ""
      if first then
        s += "if(#{condition}){"
      else
        if condition then
          s += "} else if(#{condition}){"
        else
          s += "} else {"
        end
      end
      return s
    end
    def print(*args)
      s=""
      s += " "*BOAST::get_indent_level
      s += self.to_str(@conditions.first)
      BOAST::increment_indent_level      
      BOAST::get_output.puts s
      if @blocks.size > 0 then
        if @blocks[0] then
          @blocks[0].call(*args)
        end
        @blocks[1..-1].each_index { |indx|
          BOAST::decrement_indent_level      
          s=""
          s += " "*BOAST::get_indent_level 
          s += self.to_str(@conditions[1..-1][indx],false)
          BOAST::increment_indent_level
          BOAST::get_output.puts s
          @blocks[1..-1][indx].call(*args)
        }
        self.close
      end
      return self
    end
    def close(final=true)
      return self.close_fortran(final) if BOAST::get_lang == FORTRAN
      return self.close_c(final) if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def close_c(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "}"
      BOAST::get_output.puts s if final
      return s
    end
    def close_fortran(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "end if"
      BOAST::get_output.puts s if final
      return s
    end

  end
 
  class For
    attr_reader :iterator
    attr_reader :begin
    attr_reader :end
    attr_reader :step

    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    def initialize(i, b, e, s=1, &block)
      @iterator = i
      @begin = b
      @end = e
      @step = s
      @block = block
    end
    def to_s
      self.to_str
    end
    def to_str
      return self.to_str_fortran if BOAST::get_lang == FORTRAN
      return self.to_str_c if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def to_str_fortran
      s = ""
      s += "do #{@iterator}=#{@begin}, #{@end}"
      s += ", #{@step}" if 1 != @step
      return s
    end
    def to_str_c
      s = ""
      s += "for(#{@iterator}=#{@begin}; #{@iterator}<=#{@end}; #{@iterator}+=#{@step}){"
      return s
    end

    def unroll(*args)
      raise "Block not given!" if not @block
      BOAST::push_env( :replace_constants => true )
      begin
        if @begin.kind_of?(Variable) then
          start = @begin.constant
        elsif @begin.kind_of?(Expression) then
          start = eval "#{@begin}"
        else
          start = @begin.to_i
        end
        if @end.kind_of?(Variable) then
          e = @end.constant
        elsif @end.kind_of?(Expression) then
          e = eval "#{@end}"
        else
          e = @end.to_i
        end
        if @step.kind_of?(Variable) then
          step = @step.constant
        elsif @step.kind_of?(Expression) then
          step = eval "#{@step}"
        else
          step = @step.to_i
        end
        raise "Invalid bounds (not constants)!" if not ( start and e and step )
      rescue Exception => ex
        if not ( start and e and step ) then
          BOAST::pop_env( :replace_constants )
          return self.print(*args) if not ( start and e and step )
        end
      end
      BOAST::pop_env( :replace_constants )
      range = start..e
      @iterator.force_replace_constant = true
      range.step(step) { |i|
        @iterator.constant = i
        @block.call(*args)
      }
      @iterator.force_replace_constant = false
      @iterator.constant = nil
    end

    def print(*args)
      final = true
      s=""
      s += " "*BOAST::get_indent_level if final
      s += self.to_str
      BOAST::increment_indent_level      
      BOAST::get_output.puts s if final
      if @block then
        s += "\n"
        @block.call(*args)
        s += self.close
      end
      return s
    end

    def close(final=true)
      return self.close_fortran(final) if BOAST::get_lang == FORTRAN
      return self.close_c(final) if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def close_c(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "}"
      BOAST::get_output.puts s if final
      return s
    end
    def close_fortran(final=true)
      s = ""
      BOAST::decrement_indent_level      
      s += " "*BOAST::get_indent_level if final
      s += "enddo"
      BOAST::get_output.puts s if final
      return s
    end
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

