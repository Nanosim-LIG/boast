module BOAST
  class Index < Expression
    include BOAST::Inspectable
    attr_reader :source
    attr_reader :indexes
    def initialize(source, indexes)
      @source = source
      @indexes = indexes
    end

    def to_var
      var = @source.copy("#{self}", :const => nil, :constant => nil, :dim => nil, :dimension => nil, :direction => nil, :dir => nil)
      return var
    end

    def to_s
      return self.to_s_fortran if BOAST::get_lang == FORTRAN
      return self.to_s_c if [C, CL, CUDA].include?( BOAST::get_lang )
    end

    def to_s_fortran
      s = ""
      s += "#{@source}(#{@indexes.join(", ")})"
      return s
    end

    def to_s_texture
      raise "Unsupported language #{BOAST::get_lang} for texture!" if not [CL, CUDA].include?( BOAST::get_lang )
      raise "Write is unsupported for textures!" if not ( @source.constant or @source.direction == :in )
      dim_number = 1
      if @source.dimension then
        dim_number == @source.dimension.size
      end
      raise "Unsupported number of dimension: #{dim_number}!" if dim_number > 3
      s = ""
      if BOAST::get_lang == CL then
        s += "as_#{@source.type.decl}("
        s += "read_imageui(#{@source}, #{@source.sampler}, "
        if dim_number == 1 then
          s += "int2(#{@indexes[0]},0)"
        else
          if dim_number == 2 then
            s += "int2("
          else
            s += "int3("
          end
          s += "#{@indexes.join(", ")})"
        end
        s += ")"
        if @source.type.size == 4 then
          s += ".x"
        elsif @source.type.size == 8 then
          s += ".xy"
        end
        s += ")"
      else
        s += "tex#{dim_number}Dfetch(#{@source},"
        if dim_number == 1 then
          s += "#{@indexes[0]}"
        else
          if dim_number == 2 then
            s += "int2("
          else
            s += "int3("
          end
          s += "#{@indexes.join(", ")})"
        end
        s += ")"
      end
      return s
    end

    def to_s_c
      return to_s_texture if @source.texture
      dim = @source.dimension.first
      if dim.val2 then
        start = dim.val1
      else
        start = BOAST::get_array_start
      end
      sub = "#{@indexes.first} - (#{start})"
      i=1
      ss = ""
      @source.dimension[0..-2].each{ |d|
        if d.size then
          ss += " * (#{d.size})"
        elsif d.val2 then
          ss += " * (#{d.val2} - (#{d.val1}) + 1)"
        else
          raise "Unkwown dimension size!"
        end
        dim = @source.dimension[i]
        if dim.val2 then
          start = dim.val1
        else
          start = BOAST::get_array_start
        end
        sub += " + (#{@indexes[i]} - (#{start}))"+ss
        i+=1
      }
      if BOAST::get_replace_constants then
        begin
#         puts sub
         indx = eval(sub)
         indx = indx.to_i
#         puts indx
         return "#{@source.constant[indx]}"
        rescue Exception => e
        end
      end
      s = "#{@source}[" + sub + "]"
      return s
    end

    def print
      s=""
      s += BOAST::indent
      s += self.to_s
      s += ";" if [C, CL, CUDA].include?( BOAST::get_lang )
      BOAST::get_output.puts s
      return self
    end

  end

end
