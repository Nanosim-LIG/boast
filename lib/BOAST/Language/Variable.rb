module BOAST

  # @!parse module Functors; functorize Dimension; end
  class Dimension
    include PrivateStateAccessor
    include Inspectable
    extend Functor

    attr_reader :val1
    attr_reader :val2
    attr_reader :size

    # Creates a new {Dimension}.
    # @overload initialize()
    #   Creates a {Dimension} of unknown {#size}, used to declare an array of unknown size.
    # @overload initialize( size )
    #   Creates a {Dimension} of size *size*, {#start} is computed at evaluation as {BOAST.get_array_start}.
    #   @param [Object] size can be an integer or a {Variable} or {Expression}
    # @overload initialize( lower, upper )
    #   Creates a {Dimension} with a lower and upper bound. {#size} is computed as 'upper - lower + 1' and can be an {Expression}
    #   @param [Object] lower bound of the {Dimension}
    #   @param [Object] upper bound of the {Dimension}
    def initialize(v1=nil,v2=nil)
      if v1 then
        if v2 then
          begin
            @size = v2-v1+1
          rescue
            @size = Expression::new(Substraction, v2, v1) + 1
          end
        else
          @size = v1
        end
      else
        @size = nil
      end
      @val1 = v1
      @val2 = v2
    end

    # Returns a String representation of the {Dimension}
    def to_s
      if lang == FORTRAN and @val2 then
        return "#{@val1}:#{@val2}"
      elsif lang == FORTRAN and size.nil?
        return "*"
      elsif lang == FORTRAN and get_array_start != 1 then
        return "#{get_array_start}:#{@size-(1+get_array_start)}"
      else
        return @size.to_s
      end 
    end

    # Returns the start of the {Dimension} as given at initialization or as computed {BOAST.get_array_start}.
    def start
      if @val2 then
        return @val1
      else
        return get_array_start
      end
    end

    # Returns the end of the {Dimension} if the size is known.
    def finish
      if @val2 then
        return @val2
      elsif @size
        if 0.equal?(get_array_start) then
          return @size - 1
        else
          if 1.equal?(get_array_start) then
            return @size
          else
            begin
              return @size + get_array_start - 1
            rescue
              return Expression::new(Addition, @size, get_array_start) - 1
            end
          end
        end
      else
        return nil
      end
    end

  end

  module Functors
    alias Dim Dimension
  end

  class ConstArray < Array
    include PrivateStateAccessor
    include Inspectable

    attr_accessor :shape

    def initialize(array,type = nil)
      super(array)
      @type = type::new if type
    end

    def to_s
      return to_s_fortran if lang == FORTRAN
      return to_s_c if [C, CL, CUDA].include?( lang )
    end

    private

    def to_s_fortran
      arr = flatten
      s = ""
      return s if arr.first.nil?
      s += "reshape(" if @shape
      s += "(/ &\n"
      s += arr.first.to_s
      s += @type.suffix if @type
      arr[1..-1].each { |v|
        s += ", &\n"+v.to_s
        s += @type.suffix if @type
      }
      s += " /)"
      s += ", shape(#{@shape}))" if @shape
      return s
    end

    def to_s_c
      arr = flatten
      s = ""
      return s if arr.first.nil?
      s += "{\n"
      s += arr.first.to_s
      s += @type.suffix if @type 
      arr[1..-1].each { |v|
        s += ",\n"+v.to_s
        s += @type.suffix if @type
      }
      s += "}"
    end

  end

  # @!parse module Functors; functorize Variable; end
  class Variable
    include PrivateStateAccessor
    include Arithmetic
    include Inspectable
    extend Functor
    include Annotation
    ANNOTATIONS = [ :name, :type, :dimension ]

    alias_method :orig_method_missing, :method_missing

    def method_missing(m, *a, &b)
      if @type.methods.include?(:members) and @type.members[m.to_s] then
        return struct_reference(type.members[m.to_s])
      elsif @type.methods.include?(:vector_length) and @type.vector_length > 1 and m.to_s[0] == 's' and lang == CL then
        required_set = m.to_s[1..-1].chars.to_a
        existing_set = [*('0'..'9'),*('a'..'z')].first(@type.vector_length)
        if required_set.length == required_set.uniq.length and (required_set - existing_set).empty? then
          return copy(name+"."+m.to_s, :vector_length => m.to_s[1..-1].length)
        else
          return orig_method_missing(m, *a, &b)
        end
      else
        return orig_method_missing(m, *a, &b)
      end
    end

    attr_reader :name
    attr_accessor :direction
    attr_accessor :constant
    attr_reader :allocate
    attr_reader :type
    attr_reader :dimension
    attr_reader :local
    attr_reader :texture
    attr_reader :sampler
    attr_reader :restrict
    attr_reader :deferred_shape
    attr_reader :optional
    attr_accessor :reference
    attr_accessor :alignment
    attr_accessor :replace_constant
    attr_accessor :force_replace_constant

    def constant?
      !!@constant
    end

    def allocate?
      !!@allocate
    end

    def texture?
      !!@texture
    end

    def local?
      !!@local
    end

    def restrict?
      !!@restrict
    end

    def replace_constant?
      !!@replace_constant
    end

    def force_replace_constant?
      !!@force_replace_constant
    end

    def dimension?
      !!@dimension
    end

    def scalar_output?
      !!@scalar_output
    end

    def optional?
      !!@optional
    end

    def align?
      !!@alignment
    end

    def deferred_shape?
      !!@deferred_shape
    end

    def reference?
      !!@reference
    end

    # Creates a new {Variable}
    # @param [#to_s] name
    # @param [DataType] type
    # @param [Hash] properties a set of named properties. Properties are also propagated to the {DataType}.
    # @option properties [Symbol] :direction (or *:dir*) can be one of *:in*, *:out* or *:inout*. Specify the intent of the variable.
    # @option properties [Array<Dimension>] :dimension (or *:dim*) variable is an array rather than a scalar. Dimensions are given in Fortran order (contiguous first).
    # @option properties [Object] :constant (or *:const*) states that the variable is a constant and give its value. For Variable with the *:dimension* property set must be a {ConstArray}
    # @option properties [Boolean] :restrict specifies that the compiler can assume no aliasing to this array.
    # @option properties [Boolean] :reference specifies that this variable is passed by reference.
    # @option properties [Symbol] :allocate specify that the variable is to be allocated and where. Can only be *:heap* or *:stack* for now.
    # @option properties [Boolean] :local indicates that the variable is to be allocated on the __local space of OpenCL devices or __shared__ space of CUDA devices. In C or FORTRAN this has the same effect as *:allocate* => *:stack*.
    # @option properties [Boolean] :texture for OpenCL and CUDA. In OpenCL also specifies that a sampler has to be generated to access the array variable.
    # @option properties [Integer] :align specifies the alignment the variable will be declared/allocated with if allocated or is supposed to have if it is coming from another context.
    # @option properties [Boolean] :replace_constant specifies that for scalar constants this variable should be replaced by its constant value. For constant arrays, the value of the array will be replaced if the index can be determined at evaluation.
    # @option properties [Boolean] :deferred_shape for Fortran interface generation mainly see Fortran documentation
    # @option properties [Boolean] :optional for Fortran interface generation mainly see Fortran documentation
    def initialize(name, type, properties={})
      @name = name.to_s
      @direction = properties[:direction] ? properties[:direction] : properties[:dir]
      @constant = properties[:constant] ? properties[:constant]  : properties[:const]
      @dimension = properties[:dimension] ? properties[:dimension] : properties[:dim]
      @local = properties[:local] ? properties[:local] : properties[:shared]
      @texture = properties[:texture]
      @allocate = properties[:allocate]
      @restrict = properties[:restrict]
      @alignment = properties[:align]
      @deferred_shape = properties[:deferred_shape]
      @optional = properties[:optional]
      @reference = properties[:reference]
      @force_replace_constant = false
      if not properties[:replace_constant].nil? then
        @replace_constant = properties[:replace_constant]
      else
        @replace_constant = true
      end
      if @texture and lang == CL then
        @sampler = Variable::new("sampler_#{name}", CustomType,:type_name => "sampler_t" ,:replace_constant => false, :constant => "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST")
      else
        @sampler = nil
      end
      @type = type::new(properties)
      @properties = properties
      if (@direction == :out or @direction == :inout) and not dimension? then
        @scalar_output = true
      else
        @scalar_output = false
      end
      @dimension = [@dimension].flatten if dimension?
    end

    def copy(name=nil,properties={})
      name = @name if not name
      h = @properties.clone
      properties.each { |k,v|
        h[k] = v
      }
      return Variable::new(name, @type.class, h)
    end

    def set_align(align)
      @alignment = align
      return self
    end

    def self.from_type(name, type, properties={})
      hash = type.to_hash
      properties.each { |k,v|
        hash[k] = v
      }
      hash[:direction] = nil
      hash[:dir] = nil
      return Variable::new(name, type.class, hash)
    end
  
    def to_s
      if force_replace_constant? or ( replace_constant? and constant? and replace_constants? and not dimension? ) then
        s = @constant.to_s + @type.suffix
        return s
      end
      if @scalar_output or @reference and [C, CL, CUDA].include?( lang ) and not decl_module? then
        return "(*#{name})"
      end
      return @name
    end

    def to_var
      return self
    end

    def set(x)
      return self === Set(x,self)
    end

    def dereference
      return copy("*(#{name})", :dimension => nil, :dim => nil, :direction => nil, :dir => nil) if [C, CL, CUDA].include?( lang )
      return Index::new(self, *(@dimension.collect { |d| d.start } ) ) if lang == FORTRAN
    end
   
    def struct_reference(x)
      return x.copy(name+"."+x.name) if [C, CL, CUDA].include?( lang )
      return x.copy(name+"%"+x.name) if lang == FORTRAN
    end
 
    def inc
      return Expression::new("++",self,nil)
    end

    def [](*args)
      return Index::new(self,*args)
    end
 
    def boast_header(lang=C)
      return decl_texture_s if texture?
      s = ""
      s += "const " if constant? or @direction == :in
      s += @type.decl
      if dimension? then
        s += " *" unless (use_vla? and lang != FORTRAN)
      end
      if not dimension? and ( lang == FORTRAN or @direction == :out or @direction == :inout or @reference ) then
        s += " *"
      end
      s += " #{@name}"
      if dimension? and use_vla? and lang != FORTRAN  then
        s += "["
        s += @dimension.reverse.collect { |d|
          d.to_s
        }.join("][")
        s += "]"
      end
      return s
    end

    def decl_ffi(alloc, lang)
      return :pointer if lang == FORTRAN and not alloc
      return :pointer if dimension?
      return :pointer if @direction == :out or @direction == :inout or @reference and not alloc
      return @type.decl_ffi
    end

    def decl
      return decl_fortran if lang == FORTRAN
      return decl_c if [C, CL, CUDA].include?( lang )
    end

    def align
      if dimension? then
        if align? or default_align > 1 then
          a = ( align? ? alignment : 1 )
          a = ( a >= default_align ? a : default_align )
          return align_c(a) if lang == C
          return align_fortran(a) if lang == FORTRAN
        end
      end
      return nil
    end

    def alloc( dims = nil, align = get_address_size )
      @dimension = [dims].flatten if dims
      dims = @dimension
      raise "Cannot allocate array with unknown dimension!" unless dims
      return alloc_fortran(dims) if lang == FORTRAN
      return alloc_c(dims, align) if lang == C
    end

    def dealloc
      return dealloc_fortran if lang == FORTRAN
      return dealloc_c if lang == C
    end

    private

    def __const?
      return !!( constant? or @direction == :in )
    end

    def __global?
      return !!( lang == CL and @direction and dimension? and not (@properties[:register] or @properties[:private] or local?) )
    end

    def __local?
      return !!( lang == CL and local? )
    end

    def __shared?(device = false)
      return !!( lang == CUDA and local? and not device )
    end

    def __vla_array?
      return !!( use_vla? and dimension? and not decl_module? )
    end

    def __pointer_array?(device = false)
      return !!( dimension? and not constant? and not ( allocate? and @allocate != :heap ) and (not local? or (local? and device)) )
    end

    def __pointer?(device = false)
      return !!( ( not dimension? and ( @direction == :out or @direction == :inout or @reference ) ) or __pointer_array?(device) )
    end

    def __restrict?
      return !!( restrict? and not decl_module? )
    end

    def __dimension?(device = false)
      return !!( dimension? and ((local? and not device) or ( ( allocate? and @allocate != :heap ) and not constant?)) )
    end

    def __align?
      return !!( dimension? and (align? or default_align > 1) and (constant? or (allocate? and @allocate != :heap ) ) )
    end

    def decl_c_s(device = false)
      return decl_texture_s if texture?
      s = ""
      s += "const " if __const?
      s += "__global " if __global?
      s += "__local " if __local?
      s += "__shared__ " if __shared?(device)
      s += @type.decl
      if __vla_array? then
        s += " #{@name}["
        s += "__restrict__ " if __restrict?
        s += @dimension.reverse.collect { |d|
          d.to_s
        }.join("][")
        s += "]"
      else
        s += " *" if __pointer?(device)
        if __pointer_array?(device) and __restrict? then
          if lang == CL
            s += " restrict"
          else
            s += " __restrict__" unless use_vla?
          end
        end
        s += " #{@name}"
        if dimension? and constant? then
          s += "[]"
        end
        if __dimension?(device) then
          s +="[("
          s += @dimension.collect{ |d| d.to_s }.reverse.join(")*(")
          s +=")]"
        end 
      end
      if __align? and lang != CUDA then
        a = ( align? ? alignment : 1 )
        a = ( a >= default_align ? a : default_align )
        s+= " __attribute((aligned(#{a})))"
      end
      s += " = #{@constant}" if constant?
      return s
    end

    def decl_texture_s
      raise LanguageError, "Unsupported language #{lang} for texture!" if not [CL, CUDA].include?( lang )
      raise "Write is unsupported for textures!" if not (constant? or @direction == :in)
      dim_number = 1
      if dimension? then
        dim_number == @dimension.size
      end
      raise "Unsupported number of dimension: #{dim_number}!" if dim_number > 3
      s = ""
      if lang == CL then
        s += "__read_only "
        if dim_number < 3 then
          s += "image2d_t " #from OCL 1.2+ image1d_t is defined
        else
          s += "image3d_t "
        end
      else
        s += "texture<#{@type.decl}, cudaTextureType#{dim_number}D, cudaReadModeElementType> "
      end
      s += @name
      return s
    end

    def decl_c
      s = ""
      s += indent
      s += decl_c_s
      s += finalize
      output.print s
      return self
    end

    def align_c(a)
      return FuncCall::new("__assume_aligned", @name, a)
    end

    def align_fortran(a)
      return Pragma::new("DIR", "ASSUME_ALIGNED", "#{@name}: #{a}")
    end

    def alloc_fortran( dims = nil )
      return FuncCall::new(:allocate, FuncCall(name, * dims ) )
    end

    def alloc_c( dims = nil, align = get_address_size)
      d = dims.collect { |d| d.to_s }.reverse.join(")*(")
      if align > (OS.bits/8) then
        # check alignment is a power of 2
        raise "Invalid alignment #{align}!" if align & (align - 1) != 0
        return FuncCall::new(:posix_memalign, address, align, FuncCall::new(:sizeof, @type.decl) * d)
      else
        return self === FuncCall::new(:malloc, FuncCall::new(:sizeof, @type.decl) * d).cast(self)
      end
    end

    def dealloc_fortran
      return FuncCall::new(:deallocate, self)
    end

    def dealloc_c
      return FuncCall::new(:free, self)
    end

    def decl_fortran
      s = ""
      s += indent
      s += @type.decl
      s += ", intent(#{@direction})" if @direction
      s += ", optional" if optional?
      s += ", allocatable" if allocate? and @allocate == :heap
      s += ", parameter" if constant?
      if dimension? then
        s += ", dimension("
        s += @dimension.collect { |d|
          if deferred_shape? or ( allocate? and @allocate == :heap )
            ":"
          else
            d.to_s
          end
        }.join(", ")
        s += ")"
      end
      s += " :: #{@name}"
      if constant? then
        @constant.shape = self if dimension? and @constant.kind_of?(ConstArray)
        s += " = #{@constant}"
        s += @type.suffix if not dimension? and @type
      end
      s += finalize
      output.print s
      if dimension? and (align? or default_align > 1) and (constant? or ( allocate? and @allocate != :heap ) ) then
        a = ( align? ? alignment : 1 )
        a = ( a >= default_align ? a : default_align )
        s = ""
        s += indent
        s += "!DIR$ ATTRIBUTES ALIGN: #{a}:: #{name}"
        s += finalize
        output.print s
      end
      return self
    end

    def finalize
       s = ""
       s += ";" if [C, CL, CUDA].include?( lang )
       s+="\n"
       return s
    end

  end

  module Functors
    alias Var Variable
  end

end
