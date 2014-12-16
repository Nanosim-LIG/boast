module BOAST

  class Slice
    include PrivateStateAccessor
    include Inspectable
    attr_reader :source
    attr_reader :slices

    def initialize(source, *slices)
      raise "Cannot slice a non array Variable!" if not source.dimension?
      raise "Invalid slice!" if slices.length != source.dimension.length
      @source = source
      @slices = slices
    end

    def to_s_c
      s = "#{@source}["
      dims = @source.dimension.reverse
      slices = @slices.reverse
      slices_to_c = []
      slices.each_index { |indx|
        slice = slices[indx]
        if slice.kind_of?(Array) then
          if slice.length == 0 then
            slices_to_c.push(":")
          elsif slice.length == 1 then
            slices_to_c.push("#{Expression::new(Substraction, slice[0], dims[indx].start)}")
          else
            start, finish, step = slice
            start_c = Expression::new(Substraction, start, dims[indx].start)
            length = Expression::new(Substraction, finish, start) + 1
            slices_to_c.push("#{start_c}:#{length}")
          end
        elsif slice
          slices_to_c.push("#{Expression::new(Substraction, slice, dims[indx].start)}")
        else
          slices_to_c.push(":")
        end
      }
      return s + slices_to_c.join("][") + "]"
    end

    def to_s_fortran
      slices_to_fortran = @slices.collect { |slice|
        if slice then
          sl = [slice].flatten
          if sl.length > 0 then
            sl.join(":")
          else
            ":"
          end
        else
          ":"
        end
      }
      return "#{source}(#{slices_to_fortran.join(",")})"
    end

    def to_s
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

  end

  class Variable

    def slice(*slices)
      Slice::new(self, *slices)
    end

  end

end
