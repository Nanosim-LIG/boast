module BOAST

  class Index < Expression
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
      return to_s_fortran if lang == FORTRAN
      return to_s_c if [C, CL, CUDA].include?( lang )
    end

    def to_s_fortran
      s = ""
      s += "#{@source}(#{@indexes.join(", ")})"
      return s
    end

    def to_s_texture
      raise "Unsupported language #{lang} for texture!" if not [CL, CUDA].include?( lang )
      raise "Write is unsupported for textures!" if not ( @source.constant or @source.direction == :in )
      dim_number = 1
      if @source.dimension then
        dim_number == @source.dimension.size
      end
      raise "Unsupported number of dimension: #{dim_number}!" if dim_number > 3
      s = ""
      if lang == CL then
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

    def to_s_c_reversed
      indxs = @indexes.reverse
      dims = @source.dimension.reverse
      ss = nil
      (0...dims.length).each { |indx|
        s = ""
        dim = dims[indx]
        s += "#{indxs[indx]}"
        if dim.val2 then
          s += " - (#{dim.val1})"
        elsif 0 != get_array_start then
          s += " - (#{get_array_start})"
        end
        if ss then
          if dim.size then
            s += " + (#{dim.size}) * "
          elsif dim.val2 then
            s += " + (#{dim.val2} - (#{dim.val1}) + 1) * "
          else
            raise "Unkwown dimension size!"
          end
          s += "(#{ss})"
        end
        ss = s
      }
      return ss
    end

    def to_s_c
      return to_s_texture if @source.texture
      sub = to_s_c_reversed
      if get_replace_constants then
        begin
         indx = eval(sub)
         indx = indx.to_i
         return "#{@source.constant[indx]}"
        rescue Exception => e
        end
      end
      s = "#{@source}[" + sub + "]"
      return s
    end

    def pr
      s=""
      s += indent
      s += to_s
      s += ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

  end

end
