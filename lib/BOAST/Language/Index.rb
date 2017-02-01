module BOAST

  class Index < Expression
    attr_reader :source
    attr_reader :indexes
    attr_accessor :alignment
    attr_accessor :vector_index

    def method_missing(m, *a, &b)
      var = to_var
      if var.type.methods.include?(:members) and var.type.members[m.to_s] then
        return struct_reference(type.members[m.to_s])
      elsif var.vector? and m.to_s[0] == 's' and lang != CUDA then
        required_set = m.to_s[1..-1].chars.to_a
        existing_set = [*('0'..'9'),*('a'..'z')].first(var.type.vector_length)
        if required_set.length == required_set.uniq.length and (required_set - existing_set).empty? then
          return var.copy(var.name+"."+m.to_s, :vector_length => m.to_s[1..-1].length) if lang == CL
          @vector_index = existing_set.index(required_set[0])
          return self
        else
          return super
        end
      else
        return super
      end
    end

    def initialize(source, *indexes)
      @source = source
      @indexes = indexes
      @vector_index = nil
    end

    def align?
      return !!@alignment
    end

    def set_align(align)
      @alignment = align
      return self
    end

    def to_var
      options = { :const => nil, :constant => nil, :dim => nil, :dimension => nil, :direction => nil, :dir => nil, :align => alignment }
      options[:vector_length] = 1 if @vector_index
      var = @source.copy("#{self}", options)
      return var
    end

    def set(x)
      return to_var === Set(x,to_var)
    end

    def copy(*args)
      return to_var.copy(*args)
    end

    def type
      return to_var.type
    end

    def to_s
      if get_replace_constants and @source.constant? then
        begin
          const = @source.constant
          indxs = @indexes.reverse
          dims = @source.dimension.reverse
          (0...dims.length).each { |indx|
            dim = dims[indx]
            s = "#{indxs[indx]}"
            s += " - (#{dim.start})" unless 0.equal?(dim.start)
            ind = eval(s)
            ind = ind.to_i
            const = const[ind]
          }
          return "#{const}#{@source.type.suffix}"
        rescue Exception
        end
      end
      return to_s_fortran if lang == FORTRAN
      return to_s_c if [C, CL, CUDA].include?( lang )
    end

    def pr
      s=""
      s += indent
      s += to_s
      s += ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

    private

    def to_s_fortran
      indexes_dup = []
      @source.dimension.each_with_index { |d,i|
        if d.size.nil? and get_array_start != 1 then
           indexes_dup.push( (@indexes[i] - d.start + 1).to_s )
        else
           indexes_dup.push( (@indexes[i]).to_s )
        end
      }
      s = ""
      s += "#{@source}(#{@source.vector? ? (@vector_index ? "#{@vector_index+1}, " : ":, ") : "" }#{indexes_dup.join(", ")})"
      return s
    end

    def to_s_texture
      raise LanguageError, "Unsupported language #{lang} for texture!" if not [CL, CUDA].include?( lang )
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

    def to_s_use_vla
      indxs = @indexes.reverse
      dims = @source.dimension.reverse
      t = (0...dims.length).collect { |indx|
        s = "#{indxs[indx]}"
        dim = dims[indx]
        s += " - (#{dim.start})" unless 0.equal?(dim.start)
        s
      }
      return t.join("][")
    end

    def to_s_c_reversed
      indxs = @indexes.reverse
      dims = @source.dimension.reverse
      ss = nil
      (0...dims.length).each { |indx|
        s = ""
        dim = dims[indx]
        s += "#{indxs[indx]}"
        s += " - (#{dim.start})" unless 0.equal?(dim.start)
        if ss then
          if dim.size then
            s += " + (#{dim.size}) * "
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
      if use_vla? then
        sub = to_s_use_vla
      else
        sub = to_s_c_reversed
      end
      s = "#{@source}[" + sub + "]"
      if @vector_index then
        s += "[#{@vector_index}]"
      end
      return s
    end

  end

end
