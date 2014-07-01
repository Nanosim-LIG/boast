module BOAST
  class Variable
    include BOAST::Arithmetic

    alias_method :orig_method_missing, :method_missing

    def method_missing(m, *a, &b)
      return self.struct_reference(type.members[m.to_s]) if type.members[m.to_s]
#      return self.get_element(m.to_s) if type.getters[m.to_s]
#      return self.set_element(m.to_s) if type.setters[m.to_s]
      return self.orig_method_missing(m, *a, &b)
    end

    def self.parens(*args,&block)
      return self::new(*args,&block)
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
      if @texture and BOAST::get_lang == CL then
        @sampler = Variable::new("sampler_#{name}", BOAST::CustomType,:type_name => "sampler_t" ,:replace_constant => false, :constant => "CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST")
      else
        @sampler = nil
      end
      @type = type::new(hash)
      @hash = hash
    end

    def copy(name=@name,options={})
      hash = @hash.clone
      options.each { |k,v|
        hash[k] = v
      }
      return Variable::new(name, @type.class, hash)
    end

    def Variable.from_type(name, type, options={})
      hash = type.to_hash
      options.each { |k,v|
        hash[k] = v
      }
      return Variable::new(name, type.class, hash)
    end
  
    def to_s
      self.to_str
    end    

    def to_str
      if @force_replace_constant or ( @replace_constant and @constant and BOAST::get_replace_constants and not @dimension ) then
        s = @constant.to_s 
        s += "_wp" if BOAST::get_lang == FORTRAN and @type and @type.size == 8
        return s
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
      return self.copy("*(#{self.name})", :dimension => false, :dim => false) if [C, CL, CUDA].include?( BOAST::get_lang )
      return self if BOAST::get_lang == FORTRAN
      #return Expression::new("*",nil,self)
    end
   
    def struct_reference(x)
      return x.copy(self.name+"."+x.name) if [C, CL, CUDA].include?( BOAST::get_lang )
      return x.copy(self.name+"%"+x.name) if BOAST::get_lang == FORTRAN
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
       s += ";" if [C, CL, CUDA].include?( BOAST::get_lang )
       s+="\n"
       return s
    end

    def decl_c(final=true, device=false)
      return decl_texture(final) if @texture
      s = ""
      s += self.indent if final
      s += "const " if @constant or @direction == :in
      s += "__global " if @direction and @dimension and not (@hash[:register] or @hash[:private] or @local) and BOAST::get_lang == CL
      s += "__local " if @local and BOAST::get_lang == CL
      s += "__shared__ " if @local and not device and BOAST::get_lang == CUDA
      s += @type.decl
      if(@dimension and not @constant and not @allocate and (not @local or (@local and device))) then
        s += " *"
        if @restrict then
          if BOAST::get_lang == CL
            s += " restrict"
          else
            s += " __restrict__"
          end
        end
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
      s += self.finalize if final
      BOAST::get_output.print s if final
      return s
    end

    def header(lang=C,final=true)
      return decl_texture(final) if @texture
      s = ""
      s += self.indent if final
      s += "const " if @constant or @direction == :in
      s += "__global " if @direction and @dimension and BOAST::get_lang == CL
      s += "__local " if @local and BOAST::get_lang == CL
      s += "__shared__ " if @local and BOAST::get_lang == CUDA
      s += @type.decl
      if(@dimension and not @constant and not @local) then
        s += " *"
      end
      if not @dimension and lang == FORTRAN then
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

    def decl(final=true,device=false)
      return self.decl_fortran(final) if BOAST::get_lang == FORTRAN
      return self.decl_c(final, device) if [C, CL, CUDA].include?( BOAST::get_lang )
    end

    def decl_texture(final=true)
      raise "Unsupported language #{BOAST::get_lang} for texture!" if not [CL, CUDA].include?( BOAST::get_lang )
      raise "Write is unsupported for textures!" if not (@constant or @direction == :in)
      dim_number = 1
      if @dimension then
        dim_number == @dimension.size
      end
      raise "Unsupported number of dimension: #{dim_number}!" if dim_number > 3
      s = ""
      s += self.indent if final
      if BOAST::get_lang == CL then
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
      s += self.finalize if final
      BOAST::get_output.print s if final
      return s
    end


    def decl_fortran(final=true)
      s = ""
      s += self.indent if final
      s += @type.decl
      s += ", intent(#{@direction})" if @direction 
      s += ", parameter" if @constant
      if(@dimension) then
        s += ", dimension("
        dim = @dimension[0].to_str
        if dim then
          s += dim
          @dimension[1..-1].each { |d|
             s += ", "
             s += d
          }
        else
          s += "*"
        end
        s += ")"
      end
      s += " :: #{@name}"
      if @constant
        s += " = #{@constant}"
        s += "_wp" if not @dimension and @type and @type.size == 8
      end
      s += self.finalize if final
      BOAST::get_output.print s if final
      return s
    end

  end

end
