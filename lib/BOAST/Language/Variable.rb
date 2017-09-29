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
            @size = Expression::new(Subtraction, v2, v1) + 1
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
      s << "reshape(" if @shape
      s << "(/ &\n"
      s << arr.first.to_s
      s << @type.suffix if @type
      arr[1..-1].each { |v|
        s << ", &\n"+v.to_s
        s << @type.suffix if @type
      }
      s << " /)"
      s << ", shape(#{@shape}))" if @shape
      return s
    end

    def to_s_c
      arr = flatten
      s = ""
      return s if arr.first.nil?
      s << "{\n"
      s << arr.first.to_s
      s << @type.suffix if @type
      arr[1..-1].each { |v|
        s << ",\n"+v.to_s
        s << @type.suffix if @type
      }
      s << "}"
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

    def method_missing(m, *a, &b)
      if @type.kind_of?(CStruct) and @type.members[m.to_s] then
        return struct_reference(type.members[m.to_s])
      elsif vector? and m.to_s[0] == 's' and lang != CUDA then
        required_set = m.to_s[1..-1].chars.to_a
        existing_set = [*('0'..'9'),*('a'..'z')].first(@type.vector_length)
        if required_set.length == required_set.uniq.length and (required_set - existing_set).empty? then
          return copy(name+"."+m.to_s, :vector_length => m.to_s[1..-1].length) if lang == CL
          return copy("#{name}(#{existing_set.index(required_set[0])+1})", :vector_length => nil) if lang == FORTRAN
          return copy("#{name}[#{existing_set.index(required_set[0])}]", :vector_length => nil) if lang == C
          return super
        else
          return super
        end
      else
        return super
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
    attr_writer :alignment
    attr_accessor :replace_constant
    attr_accessor :force_replace_constant

    def alignment
      return @type.total_size if vector? and lang == FORTRAN and not @alignment
      return @alignment
    end

    def constant?
      @constant
    end

    def allocate?
      @allocate
    end

    def texture?
      @texture
    end

    def local?
      @local
    end

    def restrict?
      @restrict
    end

    def replace_constant?
      @replace_constant
    end

    def force_replace_constant?
      @force_replace_constant
    end

    def dimension?
      @dimension
    end

    def scalar_output?
      @scalar_output
    end

    def optional?
      @optional
    end

    def align?
      alignment
    end

    def deferred_shape?
      @deferred_shape
    end

    def reference?
      @reference
    end

    def vector?
      @type.vector?
    end

    def meta_vector?
      @type.meta_vector?
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
    # @option properties [Integer] :align specifies the alignment the variable will be declared/allocated with if allocated or is supposed to have if it is coming from another context (in bytes).
    # @option properties [Boolean] :replace_constant specifies that for scalar constants this variable should be replaced by its constant value. For constant arrays, the value of the array will be replaced if the index can be determined at evaluation.
    # @option properties [Boolean] :deferred_shape for Fortran interface generation mainly see Fortran documentation
    # @option properties [Boolean] :optional for Fortran interface generation mainly see Fortran documentation
    def initialize(name, type, properties={})
      @name = name.to_s
      @direction = properties[:direction] or @direction = properties[:dir]
      @constant = properties[:constant] or @constant = properties[:const]
      @dimension = properties[:dimension] or @dimension = properties[:dim]
      @local = properties[:local] or @local = properties[:shared]

      @texture = properties[:texture]
      @allocate = properties[:allocate]
      @restrict = properties[:restrict]
      @alignment = properties[:align]
      @deferred_shape = properties[:deferred_shape]
      @optional = properties[:optional]
      @reference = properties[:reference]
      @force_replace_constant = false
      @replace_constant = properties[:replace_constant]

      if @texture and lang == CL then
        @sampler = Variable::new("sampler_#{name}", CustomType,:type_name => "sampler_t" ,:replace_constant => false, :constant => "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST")
      else
        @sampler = nil
      end

      @scalar_output = false
      if @dimension then
        @dimension = [@dimension].flatten
      else
        @scalar_output = true if @direction == :out or @direction == :inout
      end

      @type = type::new(properties)
      @properties = properties
    end

    def copy(name=nil,properties={})
      name = @name unless name
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
      if force_replace_constant? or ( ( replace_constant? or replace_constants? ) and constant? and not dimension? ) then
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

    # Indexes a {Variable} with the :dimension (or :dim) property set
    # @param [Array{#to_s, Range, [first, last, step], :all, nil}] args one entry for each {Dimension} of the {Variable}.
    #   * Range: if an index is a Range, the result will be a {Slice}. The Range can be exclusive. The first and last item of the Range will be considered first and last index in the corresponding {Dimension}.
    #   * [first, last, step]: if an index is an Array, the result will be a {Slice}. The first and last item of the array will be considered first and last index in the corresponding {Dimension}. If a step is given the range will be iterated by step.
    #   * :all, nil: The whole dimension will be used for the slice. But indexing will start at #get_array_start instead of the original index.
    #   * #to_s: If an index is none of the above it will be considered a scalar index. If all indexes are scalar an {Index} will be returned.
    # @return [Slice, Index]
    def [](*args)
      slice = false
      args.each { |a|
        slice = true if a.kind_of?(Range) || a.kind_of?(Array) || a.kind_of?(Symbol) || a.nil?
      }
      if slice then
        return Slice::new(self, *args)
      else
        return Index::new(self, *args)
      end
    end

    def boast_header(lang=C)
      return decl_texture_s if texture?
      s = ""
      s << "const " if constant? || @direction == :in
      s << @type.decl
      if dimension? then
        s << " *" unless (use_vla? && lang != FORTRAN)
      end
      if !dimension? && ( lang == FORTRAN || @direction == :out || @direction == :inout || @reference ) then
        s << " *"
      end
      s << " #{@name}"
      if dimension? && use_vla? && lang != FORTRAN  then
        s << "["
        s << @dimension.reverse.collect(&:to_s).join("][")
        s << "]"
      end
      return s
    end

    def decl_ffi(alloc, lang)
      return :pointer if lang == FORTRAN && !alloc
      return :pointer if dimension?
      return :pointer if (@direction == :out || @direction == :inout ||  @reference) && !alloc
      return @type.decl_ffi
    end

    def decl
      return decl_fortran if lang == FORTRAN
      return decl_c if [C, CL, CUDA].include?( lang )
    end

    def align
      if dimension? then
        if align? || default_align > 1 then
          a = ( align? ? alignment : 1 )
          a = ( a >= default_align ? a : default_align )
          return align_c(a) if lang == C
          return align_fortran(a) if lang == FORTRAN
        end
      elsif vector? then
        return align_fortran(alignment) if lang == FORTRAN
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
      return ( constant? || @direction == :in )
    end

    def __global?
      return ( lang == CL && @direction && dimension? && !(@properties[:register] || @properties[:private] || local?) )
    end

    def __local?
      return ( lang == CL && local? )
    end

    def __shared?(device = false)
      return ( lang == CUDA && local? && !device )
    end

    def __vla_array?
      return ( use_vla? && dimension? && !decl_module? )
    end

    def __pointer_array?(device = false)
      return ( dimension? && !constant? && !( allocate? && @allocate != :heap ) && (!local? || (local? && device)) )
    end

    def __pointer?(device = false)
      return ( ( !dimension? && ( @direction == :out || @direction == :inout || @reference ) ) || __pointer_array?(device) )
    end

    def __restrict?
      return ( restrict? && !decl_module? )
    end

    def __dimension?(device = false)
      return ( dimension? && ((local? && !device) || ( ( allocate? && @allocate != :heap ) && !constant?)) )
    end

    def __align?
      return ( dimension? && (align? || default_align > 1) && (constant? || (allocate? && @allocate != :heap ) ) )
    end

    def __attr_align?
      return ( __align? || ( vector? && !@direction ) )
    end

    def decl_c_s(device = false)
      return decl_texture_s if texture?
      s = ""
      s << "const " if __const?
      s << "__global " if __global?
      s << "__local " if __local?
      s << "__shared__ " if __shared?(device)
      s << @type.decl
      if __vla_array? then
        s << " #{@name}["
        s << "__restrict__ " if __restrict?
        s << @dimension.reverse.collect(&:to_s).join("][")
        s << "]"
      else
        s << " *" if __pointer?(device)
        if __pointer_array?(device) && __restrict? then
          if lang == CL
            s << " restrict"
          else
            s << " __restrict__" unless use_vla?
          end
        end
        s << " #{@name}"
        if dimension? && constant? then
          s << "[]"
        end
        if __dimension?(device) then
          s << "[("
          s << @dimension.collect(&:to_s).reverse.join(")*(")
          s << ")]"
        end
      end
      if __align? && lang != CUDA then
        a = ( align? ? alignment : 1 )
        a = ( a >= default_align ? a : default_align )
        s << " __attribute((aligned(#{a})))"
      end
      s << " = #{@constant}" if constant?
      return s
    end

    def decl_texture_s
      raise LanguageError, "Unsupported language #{lang} for texture!" unless [CL, CUDA].include?( lang )
      raise "Write is unsupported for textures!" unless (constant? || @direction == :in)
      dim_number = 1
      if dimension? then
        dim_number == @dimension.size
      end
      raise "Unsupported number of dimension: #{dim_number}!" if dim_number > 3
      s = ""
      if lang == CL then
        s << "__read_only "
        if dim_number < 3 then
          s << "image2d_t " #from OCL 1.2+ image1d_t is defined
        else
          s << "image3d_t "
        end
      else
        s << "texture<#{@type.decl}, cudaTextureType#{dim_number}D, cudaReadModeElementType> "
      end
      s << @name
      return s
    end

    def decl_c
      s = ""
      s << indent
      s << decl_c_s
      s << finalize
      output.print s
      return self
    end

    def align_c(a)
      return FuncCall::new("__assume_aligned", @name, a)
    end

    def align_fortran(a)
      return Pragma::new("DIR", "ASSUME_ALIGNED", "#{@name}: #{a}")
    end

    def alloc_fortran( dims )
      dims.unshift( @type.vector_length ) if vector?
      return FuncCall::new(:allocate, FuncCall(name, * dims ) )
    end

    def alloc_c( dims, align = get_address_size)
      ds = dims.collect(&:to_s).reverse.join(")*(")
      if align > (OS.bits/8) then
        # check alignment is a power of 2
        raise "Invalid alignment #{align}!" if align & (align - 1) != 0
        return FuncCall::new(:posix_memalign, address, align, FuncCall::new(:sizeof, @type.decl) * ds)
      else
        return self === FuncCall::new(:malloc, FuncCall::new(:sizeof, @type.decl) * ds).cast(self)
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
      s << indent
      s << @type.decl
      s << ", intent(#{@direction})" if @direction
      s << ", optional" if optional?
      s << ", allocatable" if allocate? && @allocate == :heap
      s << ", parameter" if constant?
      if dimension? || vector? then
        s << ", dimension("
        if vector? then
          s << "#{@type.vector_length}"
          s << ", " if dimension?
        end
        s << @dimension.collect { |d|
          if deferred_shape? || ( allocate? && @allocate == :heap )
            ":"
          else
            d.to_s
          end
        }.join(", ") if dimension?
        s << ")"
      end
      s << " :: #{@name}"
      if constant? then
        @constant.shape = self if dimension? && @constant.kind_of?(ConstArray)
        s << " = #{@constant}"
        s << @type.suffix if !dimension? && @type
      end
      s << finalize
      output.print s
      if ( dimension? && (align? || default_align > 1) && (constant? || ( allocate? && @allocate != :heap ) ) ) || ( vector? && !@direction ) then
        a = ( align? ? alignment : 1 )
        a = ( a >= default_align ? a : default_align )
        s = ""
        s << indent
        s << "!DIR$ ATTRIBUTES ALIGN: #{a}:: #{name}"
        s << finalize
        output.print s
      end
      return self
    end

    def finalize
       s = ""
       s << ";" if [C, CL, CUDA].include?( lang )
       s << "\n"
       return s
    end

  end

  module Functors
    alias Var Variable
  end

end
