module BOAST

  class Dimension
    include BOAST::Inspectable
    extend BOAST::Functor

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

    def to_s
      s = ""
      if @val2 then
        if BOAST::get_lang == BOAST::FORTRAN then
          s += @val1.to_s
          s += ":"
          s += @val2.to_s
        elsif [BOAST::C, BOAST::CL, BOAST::CUDA].include?( BOAST::get_lang ) then
          s += (@val2 - @val1 + 1).to_s
        end
      elsif @val1.nil? then
        return nil
      else
        s += @val1.to_s
      end
      return s
    end
  end

  class ConstArray < Array
    include BOAST::Inspectable

    def initialize(array,type = nil)
      super(array)
      @type = type::new if type
    end

    def to_s
      return self.to_s_fortran if BOAST::get_lang == BOAST::FORTRAN
      return self.to_s_c if [BOAST::C, BOAST::CL, BOAST::CUDA].include?( BOAST::get_lang )
    end

    def to_s_fortran
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

    def to_s_c
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

  class Variable
    include BOAST::Arithmetic
    include BOAST::Inspectable
    extend BOAST::Functor

    alias_method :orig_method_missing, :method_missing

    def method_missing(m, *a, &b)
      return self.struct_reference(type.members[m.to_s]) if type.members[m.to_s]
#      return self.get_element(m.to_s) if type.getters[m.to_s]
#      return self.set_element(m.to_s) if type.setters[m.to_s]
      return self.orig_method_missing(m, *a, &b)
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
    attr_accessor :replace_constant
    attr_accessor :force_replace_constant

    def initialize(name,type,hash={})
      @name = name.to_s
      @direction = hash[:direction] ? hash[:direction] : hash[:dir]
      @constant = hash[:constant] ? hash[:constant]  : hash[:const]
      @dimension = hash[:dimension] ? hash[:dimension] : hash[:dim]
      @local = hash[:local] ? hash[:local] : hash[:shared]
      @texture = hash[:texture]
      @allocate = hash[:allocate]
      @restrict = hash[:restrict]
      @force_replace_constant = false
      if not hash[:replace_constant].nil? then
        @replace_constant = hash[:replace_constant]
      else
        @replace_constant = true
      end
      if @texture and BOAST::get_lang == BOAST::CL then
        @sampler = Variable::new("sampler_#{name}", BOAST::CustomType,:type_name => "sampler_t" ,:replace_constant => false, :constant => "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST")
      else
        @sampler = nil
      end
      @type = type::new(hash)
      @options = hash
      if (@direction == :out or @direction == :inout) and not @dimension then
        @scalar_output = true
      else
        @scalar_output = false
      end
      @dimension = [@dimension].flatten if @dimension
    end

    def copy(name=nil,options={})
      name = @name if not name
      h = @options.clone
      options.each { |k,v|
        h[k] = v
      }
      return Variable::new(name, @type.class, h)
    end

    def Variable.from_type(name, type, options={})
      hash = type.to_hash
      options.each { |k,v|
        hash[k] = v
      }
      hash[:direction] = nil
      hash[:dir] = nil
      return Variable::new(name, type.class, hash)
    end
  
    def to_s
      if @force_replace_constant or ( @replace_constant and @constant and BOAST::get_replace_constants and not @dimension ) then
        s = @constant.to_s 
        s += "_wp" if BOAST::get_lang == BOAST::FORTRAN and @type and @type.size == 8
        return s
      end
      if @scalar_output and [BOAST::C, BOAST::CL, BOAST::CUDA].include?( BOAST::get_lang ) then
        return "(*#{self.name})"
      end
      return @name
    end

    def to_var
      return self
    end

    def set(x)
      return Expression::new(BOAST::Set, self, x)
    end

    def dereference
      return self.copy("*(#{self.name})", :dimension => nil, :dim => nil, :direction => nil, :dir => nil) if [BOAST::C, BOAST::CL, BOAST::CUDA].include?( BOAST::get_lang )
      return self if BOAST::get_lang == BOAST::FORTRAN
      #return Expression::new("*",nil,self)
    end
   
    def struct_reference(x)
      return x.copy(self.name+"."+x.name) if [BOAST::C, BOAST::CL, BOAST::CUDA].include?( BOAST::get_lang )
      return x.copy(self.name+"%"+x.name) if BOAST::get_lang == BOAST::FORTRAN
    end
 
    def inc
      return Expression::new("++",self,nil)
    end

    def [](*args)
      return Index::new(self,args)
    end
 
    def indent
       return " "*BOAST::get_indent_level
    end

    def finalize
       s = ""
       s += ";" if [BOAST::C, BOAST::CL, BOAST::CUDA].include?( BOAST::get_lang )
       s+="\n"
       return s
    end

    def decl_c_s(device = false)
      return decl_texture_s if @texture
      s = ""
      s += "const " if @constant or @direction == :in
      s += "__global " if @direction and @dimension and not (@options[:register] or @options[:private] or @local) and BOAST::get_lang == BOAST::CL
      s += "__local " if @local and BOAST::get_lang == BOAST::CL
      s += "__shared__ " if @local and not device and BOAST::get_lang == BOAST::CUDA
      s += @type.decl
      if(@dimension and not @constant and not @allocate and (not @local or (@local and device))) then
        s += " *"
        if @restrict then
          if BOAST::get_lang == BOAST::CL
            s += " restrict"
          else
            s += " __restrict__"
          end
        end
      end
      if not @dimension and ( @direction == :out or @direction == :inout ) then
        s += " *"
      end
      s += " #{@name}"
      if @dimension and @constant then
        s += "[]"
      end
      if @dimension and ((@local and not device) or (@allocate and not @constant)) then
         s +="["
         s += @dimension.reverse.join("*")
         s +="]"
      end 
      s += " = #{@constant}" if @constant
      return s
    end

    def decl_texture_s
      raise "Unsupported language #{BOAST::get_lang} for texture!" if not [BOAST::CL, BOAST::CUDA].include?( BOAST::get_lang )
      raise "Write is unsupported for textures!" if not (@constant or @direction == :in)
      dim_number = 1
      if @dimension then
        dim_number == @dimension.size
      end
      raise "Unsupported number of dimension: #{dim_number}!" if dim_number > 3
      s = ""
      if BOAST::get_lang == BOAST::CL then
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
      s += self.indent
      s += self.decl_c_s
      s += self.finalize
      BOAST::get_output.print s
      return self
    end

    def header(lang=C,final=true)
      return decl_texture(final) if @texture
      s = ""
      s += self.indent if final
      s += "const " if @constant or @direction == :in
      s += "__global " if @direction and @dimension and BOAST::get_lang == BOAST::CL
      s += "__local " if @local and BOAST::get_lang == BOAST::CL
      s += "__shared__ " if @local and BOAST::get_lang == BOAST::CUDA
      s += @type.decl
      if(@dimension and not @constant and not @local) then
        s += " *"
      end
      if not @dimension and ( lang == BOAST::FORTRAN or @direction == :out or @direction == :inout ) then
        s += " *"
      end
      s += " #{@name}"
      if(@dimension and @constant) then
        s += "[]"
      end
      if(@dimension and @local) then
         s +="["
         s += @dimension.reverse.join("*")
         s +="]"
      end 
      s += " = #{@constant}" if @constant
      s += self.finalize if final
      BOAST::get_output.print s if final
      return s
    end

    def decl
      return self.decl_fortran if BOAST::get_lang == BOAST::FORTRAN
      return self.decl_c if [BOAST::C, BOAST::CL, BOAST::CUDA].include?( BOAST::get_lang )
    end


    def decl_fortran
      s = ""
      s += self.indent
      s += @type.decl
      s += ", intent(#{@direction})" if @direction 
      s += ", parameter" if @constant
      if(@dimension) then
        s += ", dimension("
        dim = @dimension[0].to_s
        if dim then
          s += dim
          @dimension[1..-1].each { |d|
             s += ", #{d}"
          }
        else
          s += "*"
        end
        s += ")"
      end
      s += " :: #{@name}"
      if @constant then
        s += " = #{@constant}"
        s += "_wp" if not @dimension and @type and @type.size == 8
      end
      s += self.finalize
      BOAST::get_output.print s
      return self
    end

  end

end
