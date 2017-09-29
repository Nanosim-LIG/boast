module BOAST

  class OperatorError < Error
  end

  class Operator
    extend PrivateStateAccessor
    extend Intrinsics

    DISCARD_OPTIONS = { :const => nil, :constant => nil, :direction => nil, :dir => nil, :align => nil }

    def Operator.inspect
      return "#{name}"
    end

    def Operator.convert(arg, type)
      return "#{arg}" if get_vector_name(arg.type) == get_vector_name(type) or lang == CUDA

      if arg.type.scalar? && type.vector? then
        return "#{Set::new( arg, Variable::new(:dummy, type.class, type.to_hash) )}"
      end

      return "convert_#{type.decl}( #{arg} )" if lang == CL

      path = get_conversion_path(type, arg.type)
      s = "#{arg}"
      if path.length > 1 then
        path.each_cons(2) { |slice|
          instruction = intrinsics_by_vector_name(:CVT, slice[1], slice[0])
          s = "#{instruction}( #{s} )"
        }
      end
      return s
    end

  end

  class BasicBinaryOperator < Operator

    def BasicBinaryOperator.string(arg1, arg2, return_type)
      if lang == C and (arg1.instance_of? Variable and arg2.instance_of? Variable) and (arg1.type.vector? or arg2.type.vector?) then
        instruction = intrinsics(intr_symbol, return_type.type)
        a1 = convert(arg1, return_type.type)
        a2 = convert(arg2, return_type.type)
        return "#{instruction}( #{a1}, #{a2} )"
      else
        return basic_usage( arg1, arg2 )
      end
    end

  end

  class Minus < Operator

    def Minus.string(arg1, arg2, return_type)
      return " -(#{arg2})"
    end

  end

  class Plus < Operator

    def Plus.string(arg1, arg2, return_type)
      return " +#{arg2}"
    end

  end

  class Not < Operator

    def Not.string(arg1, arg2, return_type)
      return " (.not. (#{arg2}))" if lang == FORTRAN
      return " !(#{arg2})"
    end

  end

  class Reference < Operator

    def Reference.string(arg1, arg2, return_type)
      return " #{arg2}" if lang == FORTRAN
      return " &#{arg2}"
    end

  end

  class Dereference < Operator

    def Dereference.string(arg1, arg2, return_type)
      return " *(#{arg2})"
    end

  end

  class Equal < Operator

    def Equal.string(arg1, arg2, return_type)
      return basic_usage(arg1, arg2)
    end

    def Equal.basic_usage(arg1, arg2)
      return "#{arg1} == #{arg2}"
    end

  end

  class Different < Operator

    def Different.string(arg1, arg2, return_type)
      return basic_usage(arg1, arg2)
    end

    def Different.basic_usage(arg1, arg2)
      return "#{arg1} /= #{arg2}" if lang == FORTRAN
      return "#{arg1} != #{arg2}"
    end

  end

  class Greater < Operator

    def Greater.string(arg1, arg2, return_type)
      return basic_usage(arg1, arg2)
    end

    def Greater.basic_usage(arg1, arg2)
      return "#{arg1} > #{arg2}"
    end

  end

  class Less < Operator

    def Less.string(arg1, arg2, return_type)
      return basic_usage(arg1, arg2)
    end

    def Less.basic_usage(arg1, arg2)
      return "#{arg1} < #{arg2}"
    end

  end

  class GreaterOrEqual < Operator

    def GreaterOrEqual.string(arg1, arg2, return_type)
      return basic_usage(arg1, arg2)
    end

    def GreaterOrEqual.basic_usage(arg1, arg2)
      return "#{arg1} >= #{arg2}"
    end

  end

  class LessOrEqual < Operator

    def LessOrEqual.string(arg1, arg2, return_type)
      return basic_usage(arg1, arg2)
    end

    def LessOrEqual.basic_usage(arg1, arg2)
      return "#{arg1} <= #{arg2}"
    end

  end

  class And < Operator

    def And.string(arg1, arg2, return_type)
      return basic_usage(arg1, arg2)
    end

    def And.basic_usage(arg1, arg2)
      return "#{arg1} .and. #{arg2}" if lang == FORTRAN
      return "#{arg1} && #{arg2}"
    end

  end

  class Or < Operator

    def Or.string(arg1, arg2, return_type)
      return basic_usage(arg1, arg2)
    end

    def Or.basic_usage(arg1, arg2)
      return "#{arg1} .or. #{arg2}" if lang == FORTRAN
      return "#{arg1} || #{arg2}"
    end

  end

  class Exponentiation < BasicBinaryOperator

    class << self

      def symbol
        return "**"
      end

      def intr_symbol
        return :POW
      end

      def basic_usage(arg1, arg2)
        return "(#{arg1})**(#{arg2})" if lang == FORTRAN
        return "pow(#{arg1}, #{arg2})"
      end

    end

  end

  class Multiplication < BasicBinaryOperator

    class << self

      def symbol
        return "*"
      end

      def intr_symbol
        return :MUL
      end

      def basic_usage(arg1, arg2)
        return "(#{arg1}) * (#{arg2})" 
      end
  
    end

  end

  class Addition < BasicBinaryOperator

    class << self

      def symbol
        return "+"
      end

      def intr_symbol
        return :ADD
      end
  
      def basic_usage(arg1, arg2)
        return "#{arg1} + #{arg2}" 
      end
  
    end

  end

  class Subtraction < BasicBinaryOperator

    class << self

      def symbol
        return "-"
      end

      def intr_symbol
        return :SUB
      end
  
      def basic_usage(arg1, arg2)
        return "#{arg1} - (#{arg2})" 
      end
  
    end

  end

  class Division < BasicBinaryOperator

    class << self

      def symbol
        return "/"
      end

      def intr_symbol
        return :DIV
      end
  
      def basic_usage(arg1, arg2)
        return "(#{arg1}) / (#{arg2})" 
      end
  
    end

  end

  class Min < BasicBinaryOperator

    class << self

      def intr_symbol
        return :MIN
      end

      def basic_usage(arg1, arg2)
        return "min( #{arg1}, #{arg2} )"
      end

    end

  end

  class Max < BasicBinaryOperator

    class << self

      def intr_symbol
        return :MAX
      end

      def basic_usage(arg1, arg2)
        return "max( #{arg1}, #{arg2} )"
      end

    end

  end

  class Mask
    extend Functor

    attr_reader :value
    attr_reader :length
    attr_reader :pos_values

    def empty?
      if @pos_values then
        return @pos_values == 0
      else
        return false
      end
    end

    def full?
      if @pos_values and @length
        return @pos_values == @length
      else
        return false
      end
    end

    def initialize( values, options = {} )
      length = options[:length]
      if values.kind_of?(Mask) then
        raise OperatorError, "Wrong number of mask values (#{values.length} for #{length})!" if length and values.length and values.length != length
        @value = values.value
        @length = length ? length : values.length
        @pos_values = values.pos_values
      elsif values.kind_of?(Array) then
        raise OperatorError, "Wrong number of mask values (#{values.length} for #{length})!" if length and values.length != length
        s = "0x"
        s << values.collect { |v| v != 0 ? 1 : 0 }.reverse.join
        @value = Int( s, :signed => false, :size => values.length / 8 + ( values.length % 8 > 0 ? 1 : 0 ), :constant => s )
        @length = values.length
        @pos_values = values.reject { |e| e == 0 }.length
      elsif values.kind_of?(Variable) and values.type.kind_of?(Int) then
        raise OperatorError, "Wrong mask size (#{values.type.size} for #{length / 8 + ( length % 8 > 0 ? 1 : 0 )})!" if length and values.type.size != length / 8 + ( length % 8 > 0 ? 1 : 0 )
        @value = values
        @length = length if length
      else
        raise OperatorError, "Illegal valuess for mask (#{values.class}), expecting Array of Int!"
      end
    end

    def to_s
      return @value.to_s
    end

    def to_var
      return @value
    end
  end

  # @!parse module Functors; functorize Set; end
  class Set < Operator
    extend Functor
    include Intrinsics
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :source
    attr_reader :return_type

    def initialize(source, return_type)
      @source = source
      @return_type = return_type.to_var
    end

    def type
      return @return_type.type
    end

    def to_var
      if (lang == C || lang == CL) && @return_type.type.vector? then
        if @source.kind_of?( Array ) then
          raise OperatorError,  "Invalid array length!" unless @source.length == @return_type.type.vector_length
          return @return_type.copy("(#{@return_type.type.decl})( #{@source.join(", ")} )", DISCARD_OPTIONS) if lang == CL
	  return Set(@source.first, @return_type).to_var if @source.uniq.size == 1
          begin
            instruction = intrinsics(:SET, @return_type.type)
            raise IntrinsicsError unless instruction
            eff_srcs = @source.collect { |src|
              eff_src = nil
              eff_src = src.to_var if src.respond_to?(:to_var)
              eff_src = src unless eff_src
              eff_src
            }
            return @return_type.copy("#{instruction}( #{eff_srcs.join(", ")} )",  DISCARD_OPTIONS)
          rescue IntrinsicsError
            instruction = intrinsics(:SET_LANE, @return_type.type)
            raise IntrinsicsError, "Missing instruction for SET_LANE on #{get_architecture_name}!" unless instruction
            s = Set(0, @return_type).to_s
            @source.each_with_index { |v,i|
              eff_src = nil
              eff_src = v.to_var if v.respond_to?(:to_var)
              eff_src = v unless eff_src
              s = "#{instruction}( #{eff_src}, #{s}, #{i} )"
            }
            return @return_type.copy(s, DISCARD_OPTIONS)
          end
        elsif @source.class != Variable || @source.type.scalar? then
          eff_src = nil
          eff_src = @source.to_var if @source.respond_to?(:to_var)
          eff_src = @source unless eff_src
          if lang == CL then
            return @return_type.copy("(#{@return_type.type.decl})( #{eff_src} )", DISCARD_OPTIONS) if lang == CL
          end
          if (@source.is_a?(Numeric) and @source == 0) or (@source.instance_of? Variable and @source.constant == 0) then
            begin
              instruction = intrinsics(:SETZERO, @return_type.type)
              return @return_type.copy("#{instruction}( )", DISCARD_OPTIONS) if instruction
            rescue IntrinsicsError
            end
          end
          instruction = intrinsics(:SET1, @return_type.type)
          return @return_type.copy("#{instruction}( #{eff_src} )", DISCARD_OPTIONS)
        elsif @return_type.type != @source.type
          return @return_type.copy("#{Operator.convert(@source, @return_type.type)}", DISCARD_OPTIONS)
        end
      elsif lang == FORTRAN and @return_type.type.vector? then
        if @source.kind_of?( Array ) then
          raise OperatorError,  "Invalid array length!" unless @source.length == @return_type.type.vector_length
          return "(/#{@source.join(", ")}/)"
        end
      end
      eff_src = nil
      eff_src = @source.to_var if @source.respond_to?(:to_var)
      eff_src = @source unless eff_src
      return @return_type.copy("#{eff_src}", DISCARD_OPTIONS)
    end

    def to_s
      return to_var.to_s
    end

    def pr
      s=""
      s << indent
      s << to_s
      s << ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end

  # @!parse module Functors; functorize Affectation; end
  class Affectation < Operator
    extend Functor
    include Intrinsics
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :target
    attr_reader :source
    attr_reader :options

    def initialize(target, source, options = {})
      @target = target
      @source = source
      @options = options
    end

    def type
      return target.to_var.type
    end

    def to_var
      tar = @target
      tar = @target.to_var if @target.respond_to?(:to_var)
      src = @source
      src = @source.to_var if @source.respond_to?(:to_var)
      if tar.instance_of? Variable and tar.type.vector? then
        return @target.copy("#{@target} = #{Load(@source, @target, @options)}", DISCARD_OPTIONS)
      elsif src.instance_of? Variable and src.type.vector? then
        r_t, _ = transition(tar, src, Affectation)
        opts = @options.clone
        opts[:store_type] = r_t
        return @target.copy("#{Store(@target, @source, opts)}", DISCARD_OPTIONS)
      end
      return tar.copy("#{tar ? tar : @target} = #{src ? src : @source}", DISCARD_OPTIONS)
    end

    def to_s
      return to_var.to_s
    end

    def pr
      s=""
      s << indent
      s << to_s
      s << ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end


  # @!parse module Functors; functorize Load; end
  class Load < Operator
    extend Functor
    include Intrinsics
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :source
    attr_reader :return_type
    attr_reader :options

    def initialize(source, return_type, options = {})
      @source = source
      @return_type = return_type.to_var
      @options = options
      @mask = options[:mask]
      @zero = options[:zero]
    end

    def type
      return @return_type.type
    end

    def to_var
      if lang == C or lang == CL then
        if @source.kind_of?(Array) then
          return Set(@source, @return_type).to_var
        elsif @source.instance_of? Variable or @source.respond_to?(:to_var) then
          src_var = source.to_var
          if src_var.type == @return_type.type then
            return src_var
          elsif src_var.type.scalar? then
            a2 = "#{src_var}"
            if a2[0] != "*" then
              a2 = "&" + a2
            else
              a2 = a2[1..-1]
            end
            return @return_type.copy("vload#{@return_type.type.vector_length}(0, #{a2})", DISCARD_OPTIONS) if lang == CL
            return @return_type.copy("_m_from_int64( *((int64_t * ) #{a2} ) )", DISCARD_OPTIONS) if get_architecture == X86 and @return_type.type.total_size*8 == 64
            sym = ""
            mask = nil
            mask = Mask(@mask, :length => @return_type.type.vector_length) if @mask
            if mask and not mask.full? then
              return Set(0, @return_type) if @zero and mask.empty?
              return @return_type if mask.empty?
              sym << "MASK"
              sym << "Z" if @zero
              sym << "_"
            end
            if src_var.alignment and @return_type.type.total_size and ( src_var.alignment % @return_type.type.total_size ) == 0 then
              sym << "LOADA"
            else
              sym << "LOAD"
            end
            instruction = intrinsics( sym.to_sym, @return_type.type)
            if mask and not mask.full? then
              return @return_type.copy("#{instruction}( (#{mask.value.type.decl})#{mask}, #{a2} )", DISCARD_OPTIONS) if @zero
              return @return_type.copy("#{instruction}( #{@return_type}, (#{mask.value.type.decl})#{mask}, #{a2} )", DISCARD_OPTIONS)
            end
            return @return_type.copy("#{instruction}( #{a2} )", DISCARD_OPTIONS)
          else
            return @return_type.copy("#{Operator.convert(src_var, @return_type.type)}", DISCARD_OPTIONS)
          end
        end
      elsif lang == FORTRAN then
        if @source.kind_of?(Array) then
          return Set(@source, @return_type).to_var
        elsif @source.instance_of? Variable or @source.respond_to?(:to_var) then
          if @source.to_var.type == @return_type.type then
            return @source.to_var
          elsif @source.kind_of?(Index) and @return_type.type.vector? then
            return @return_type.copy("#{Slice::new(@source.source, [@source.indexes[0], @source.indexes[0] + @return_type.type.vector_length - 1], *@source.indexes[1..-1])}", DISCARD_OPTIONS)
          end
        end
      end
      return @return_type.copy("#{@source}", DISCARD_OPTIONS)
    end

    def to_s
      return to_var.to_s
    end

    def pr
      s=""
      s << indent
      s << to_s
      s << ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end

  # @!parse module Functors; functorize MaskLoad; end
  class MaskLoad < Operator
    extend Functor
    include Intrinsics
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :source
    attr_reader :mask
    attr_reader :return_type

    def initialize(source, mask, return_type)
      @source = source
      @mask = mask
      @return_type = return_type.to_var
    end

    def get_mask
      type = @return_type.type
      return Set(@mask.collect { |m| ( m and m != 0 )  ? -1 : 0 }, Int("mask", :size => type.size, :vector_length => type.vector_length ) )
    end

    private :get_mask

    def type
      return @return_type.type
    end

    def to_var
      raise OperatorError,  "Cannot load unknown type!" unless @return_type
      type = @return_type.type
      raise LanguageError,  "Unsupported language!" unless lang == C
      raise OperatorError,  "Mask size is wrong: #{@mask.length} for #{type.vector_length}!" if @mask.length != type.vector_length
      return Load( @source, @return_type ).to_var unless @mask.include?(0)
      return Set( 0, @return_type ).to_var if @mask.uniq.size == 1 and @mask.uniq.first == 0
      instruction = intrinsics(:MASKLOAD, type)
      s = ""
      src = "#{@source}"
      if src[0] != "*" then
        src = "&" + src
      else
        src = src[1..-1]
      end
      p_type = type.copy(:vector_length => nil)
      s << "#{instruction}( (#{p_type.decl} * ) #{src}, #{get_mask} )"
      return @return_type.copy( s, DISCARD_OPTIONS)
    end

    def to_s
      return to_var.to_s
    end

    def pr
      s=""
      s << indent
      s << to_s
      s << ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end

  # @!parse module Functors; functorize Store; end
  class Store < Operator
    extend Functor
    include Intrinsics
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :dest
    attr_reader :source
    attr_reader :store_type
    attr_reader :options

    def initialize(dest, source, options = {})
      @dest = dest
      @source = source
      @store_type = options[:store_type]
      @store_type = source.to_var unless @store_type
      @mask = options[:mask]
    end

    def to_s
      if @store_type.type == @dest.type then
        return "#{@dest} = #{@source}"
      end
      if lang == C or lang == CL then
        dst = "#{@dest}"
        if dst[0] != "*" then
          dst = "&" + dst
        else
          dst = dst[1..-1]
        end
        type = @store_type.type
        return "vstore#{type.vector_length}( #{@source}, 0, #{dst} )" if lang == CL
        return "*((int64_t * ) #{dst}) = _m_to_int64( #{@source} )" if get_architecture == X86 and type.total_size*8 == 64
        sym = ""
        mask = nil
        mask = Mask(@mask, :length => @store_type.type.vector_length) if @mask
        return "" if mask and mask.empty?
        sym << "MASK_" if mask and not mask.full?
        if @dest.alignment and type.total_size and ( @dest.alignment % type.total_size ) == 0 then
          sym << "STOREA"
        else
          sym << "STORE"
        end
        instruction = intrinsics(sym.to_sym, type)
        p_type = type.copy(:vector_length => nil)
        p_type = type if get_architecture == X86 and type.kind_of?(Int)
        return "#{instruction}( (#{p_type.decl} * ) #{dst}, (#{mask.value.type.decl})#{mask}, #{@source} )" if mask and not mask.full?
        return "#{instruction}( (#{p_type.decl} * ) #{dst}, #{@source} )"
      elsif lang == FORTRAN
        if @store_type.type.vector? and @dest.kind_of?(Index) then
          return "#{Slice::new(@dest.source, [@dest.indexes[0], @dest.indexes[0] + @store_type.type.vector_length - 1], *@dest.indexes[1..-1])} = #{@source}"
        end
      end
      return "#{@dest} = #{@source}"
    end

    def pr
      s=""
      s << indent
      s << to_s
      s << ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end

  # @!parse module Functors; functorize MaskStore; end
  class MaskStore < Operator
    extend Functor
    include Intrinsics
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :dest
    attr_reader :source
    attr_reader :mask
    attr_reader :store_type

    def initialize(dest, source, mask, store_type = nil)
      @dest = dest
      @source = source
      @mask = mask
      @store_type = store_type
      @store_type = source.to_var unless @store_type
    end

    def get_mask
      type = @store_type.type
      return Set(@mask.collect { |m| ( m and m != 0 )  ? -1 : 0 }, Int("mask", :size => type.size, :vector_length => type.vector_length ) )
    end

    private :get_mask

    def to_s
      raise OperatorError,  "Cannot store unknown type!" unless @store_type
      type = @store_type.type
      raise LanguageError,  "Unsupported language!" unless lang == C
      raise OperatorError,  "Mask size is wrong: #{@mask.length} for #{type.vector_length}!" if @mask.length != type.vector_length
      return Store( @dest, @source, :store_type => @store_type ).to_s unless @mask.include?(0)
      return nil if @mask.uniq.size == 1 and @mask.uniq.first == 0
      instruction = intrinsics(:MASKSTORE, type)
      s = ""
      dst = "#{@dest}"
      if dst[0] != "*" then
        dst = "&" + dst
      else
        dst = dst[1..-1]
      end
      p_type = type.copy(:vector_length => nil)
      return s << "#{instruction}( (#{p_type.decl} * ) #{dst}, #{get_mask}, #{Operator.convert(@source, type)} )"
    end

    def pr
      ss = to_s
      if ss then
        s=""
        s << indent
        s << ss
        s << ";" if [C, CL, CUDA].include?( lang )
        output.puts s
      end
      return self
    end

  end

  # @!parse module Functors; functorize FMA; end
  class FMA < Operator
    extend Functor
    include Intrinsics
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :operand1
    attr_reader :operand2
    attr_reader :operand3
    attr_reader :return_type

    def initialize(a,b,c)
      @operand1 = a
      @operand2 = b
      @operand3 = c
      @return_type = @operand3.to_var
    end

    def convert_operand(op)
      return  "#{Operator.convert(op, @return_type.type)}"
    end

    private :convert_operand

    def type
      return @return_type.type
    end

    def to_var
      instruction = nil
      begin
        instruction = intrinsics(:FMADD,@return_type.type)
      rescue
      end
      return (@operand3 + @operand1 * @operand2).to_var unless lang != FORTRAN and @return_type and ( instruction or ( [CL, CUDA].include?(lang) ) )
      op1 = convert_operand(@operand1.to_var)
      op2 = convert_operand(@operand2.to_var)
      op3 = convert_operand(@operand3.to_var)
      if [CL, CUDA].include?(lang)
        ret_name = "fma( #{op1}, #{op2}, #{op3} )"
      else
        case architecture
        when X86
          ret_name = "#{instruction}( #{op1}, #{op2}, #{op3} )"
        when ARM
          ret_name = "#{instruction}( #{op3}, #{op1}, #{op2} )"
        else
          return (@operand3 + @operand1 * @operand2).to_var
        end
      end
      return @return_type.copy( ret_name, DISCARD_OPTIONS)
    end

    def to_s
      return to_var.to_s
    end

    def pr
      s=""
      s << indent
      s << to_s
      s << ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end

  # @!parse module Functors; functorize FMS; end
  class FMS < Operator
    extend Functor
    include Intrinsics
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :operand1
    attr_reader :operand2
    attr_reader :operand3
    attr_reader :return_type

    def initialize(a,b,c)
      @operand1 = a
      @operand2 = b
      @operand3 = c
      @return_type = @operand3.to_var
    end

    def convert_operand(op)
      return  "#{Operator.convert(op, @return_type.type)}"
    end

    private :convert_operand

    def type
      return @return_type.type
    end

    def to_var
      instruction = nil
      begin
        instruction = intrinsics(:FNMADD,@return_type.type)
      rescue
      end
      return (@operand3 - @operand1 * @operand2).to_var unless lang != FORTRAN and @return_type and ( instruction or ( [CL, CUDA].include?(lang) ) )
      op1 = convert_operand(@operand1.to_var)
      op2 = convert_operand(@operand2.to_var)
      op3 = convert_operand(@operand3.to_var)
      if [CL, CUDA].include?(lang)
        op1 = convert_operand((-@operand1).to_var)
        ret_name = "fma( #{op1}, #{op2}, #{op3} )"
      else
        case architecture
        when X86
          ret_name = "#{instruction}( #{op1}, #{op2}, #{op3} )"
        when ARM
          ret_name = "#{instruction}( #{op3}, #{op1}, #{op2} )"
        else
          return (@operand3 - @operand1 * @operand2).to_var
        end
      end
      return @return_type.copy( ret_name, DISCARD_OPTIONS)
    end

    def to_s
      return to_var.to_s
    end

    def pr
      s=""
      s << indent
      s << to_s
      s << ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end

  class Modulo < Operator
    extend Functor
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor
    include TypeTransition

    attr_reader :operand1
    attr_reader :operand2
    attr_reader :return_type

    def initialize(x,y)
      @operand1 = x
      @operand2 = y
      op1, op2 = op_to_var
      @return_type, _ = transition(op1, op2, Modulo)
    end

    def to_s
      return to_s_fortran if lang == FORTRAN
      return to_s_c if [C, CL, CUDA].include?( lang )
    end

    def pr
      s=""
      s << indent
      s << to_s
      s << ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

    def to_var
      if @return_type then
        return @return_type.copy( to_s, DISCARD_OPTIONS )
      else
        return Variable::new( to_s, get_default_type )
      end
    end

    private

    def to_s_fortran
      op1, op2 = op_to_var
      if @return_type and @return_type.type.kind_of?(Real) and ( not op1.type.kind_of?(Real) or not op2.type.kind_of?(Real) ) then
        return "modulo(real(#{op1}, #{@return_type.type.size}), #{op2})" unless op1.type.kind_of?(Real)
        return "modulo(#{op1}, real(#{op2}, #{@return_type.type.size}))"
      else
        return "modulo(#{op1}, #{op2})"
      end
    end

    def to_s_c
      op1, op2 = op_to_var
      if @return_type and @return_type.type.kind_of?(Real) then
        if @return_type.type.size <= 4 then
          return "((#{op1} < 0) ^ (#{op2} < 0) ? fmodf(#{op1}, #{op2}) + #{op2} : fmodf(#{op1}, #{op2}))"
        else
          return "((#{op1} < 0) ^ (#{op2} < 0) ? fmod(#{op1}, #{op2}) + #{op2} : fmod(#{op1}, #{op2}))"
        end
      else
        test_op1 = true
        test_op1 = false if op1.respond_to?(:type) and op1.type.respond_to?(:signed?) and not op1.type.signed?
        test_op2 = true
        test_op2 = false if op2.respond_to?(:type) and op2.type.respond_to?(:signed?) and not op2.type.signed?
        if test_op1 and test_op2 then
          return "((#{op1} < 0) ^ (#{op2} < 0) ? (#{op1} % #{op2}) + #{op2} : #{op1} % #{op2})"
        elsif test_op1 then
          return "( (#{op1} < 0) ? (#{op1} % #{op2.cast(op1)}) + #{op2} : #{op1} % #{op2})"
        elsif test_op2 then
          return "( (#{op2} < 0) ? (#{op1.cast(op2)} % #{op2}) + #{op2} : #{op1} % #{op2})"
        else
          return "(#{op1} % #{op2})"
        end
      end
    end

    def op_to_var
      op1 = @operand1.respond_to?(:to_var) ? @operand1.to_var : @operand1
      op1 = @operand1 unless op1
      op2 = @operand2.respond_to?(:to_var) ? @operand2.to_var : @operand2
      op2 = @operand2 unless op2
      return [op1, op2]
    end

  end

  # @!parse module Functors; functorize Ternary; end
  class Ternary
    extend Functor
    include Arithmetic
    include Inspectable
    include PrivateStateAccessor

    attr_reader :operand1
    attr_reader :operand2
    attr_reader :operand3
    
    def initialize(x,y,z)
      @operand1 = x
      @operand2 = y
      @operand3 = z
    end

    def to_s
      return to_s_fortran if lang == FORTRAN
      return to_s_c if [C, CL, CUDA].include?( lang )
    end

    def pr
      s=""
      s << indent
      s << to_s
      s << ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

    private

    def to_s_fortran
      op1, op2, op3 = op_to_var
      "merge(#{op2}, #{op3}, #{op1})"
    end

    def to_s_c
      op1, op2, op3 = op_to_var
      "(#{op1} ? #{op2} : #{op3})"
    end

    def op_to_var
      op1 = @operand1.respond_to?(:to_var) ? @operand1.to_var : @operand1
      op1 = @operand1 unless op1
      op2 = @operand2.respond_to?(:to_var) ? @operand2.to_var : @operand2
      op2 = @operand2 unless op2
      op3 = @operand3.respond_to?(:to_var) ? @operand3.to_var : @operand3
      op3 = @operand3 unless op3
      return [op1, op2, op3]
    end

  end

end
