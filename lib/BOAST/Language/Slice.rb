module BOAST

  class Slice < Expression
    class SliceItem
      include PrivateStateAccessor
      include Inspectable

      attr_reader :first
      attr_reader :last
      attr_reader :step
      attr_reader :length

      def all?
        @first.nil?
      end

      def scalar?
        not all? and @last.nil?
      end

      def initialize(slice)
        @first = nil
        @last = nil
        @step = nil
        if slice.kind_of?(SliceItem) then
          copy_slice!(slice)
        elsif slice.kind_of?(Range) then
          @first = slice.first
          @last = slice.last
          @length = Expression::new(Substraction, last, first)
          if slice.exclude_end? then
            @last  = Expression::new(Substraction, @last, 1)
          else
            @length = @length + 1
          end
        elsif slice.kind_of?(Array) then
          @first = slice [0] if slice.length > 0
          @last = slice [1] if slice.length > 1
          @step = slice [2] if slice.length > 2
          if @last then
            @length = Expression::new(Substraction, last, first) + 1
          end
        elsif slice.kind_of?(Symbol) then
          raise "Invalid Slice item: #{slice.inspect}!" if slice != :all
        else
          @first = slice
        end
        @length = @length / @step if @length and @step
      end

      def copy_slice!(s)
        @first = slice.first
        @last = slice.last
        @step = slice.step
        @length = slice.length
        self
      end

      def recurse!(s, d)
        if all? then
          copy_slice!(s)
        else
          if s.all? then
            @first = Expression::new(Addition, d.start, Expression::new(Substraction, @first, get_array_start) * @step )
          else
            @first = Expression::new(Addition, s.first, Expression::new(Substraction, @first, get_array_start) * @step )
          end
          if not scalar? then
            @last = @first + @length - 1
          end
        end
        return self
      end

      def to_a
        a = []
        a.push @first if @first
        a.push @last if @last
        a.push @step if @step
        return a
      end

    end

    attr_reader :source
    attr_reader :slices
    attr_accessor :alignment

    def initialize(source, *slices)
      raise "Cannot slice a non array Variable!" if not source.dimension?
      raise "Invalid slice!" if slices.length != source.dimension.length
      @source = source
      @slices = slices.collect{ |s| SliceItem::new(s) }
    end

    def dimension?
      true
    end

    def dimension
      dims = []
      slices.each_with_index { |slice, i|
        if not slice.scalar? then
          if slice.all? then
            if source.dimension[i].size then
              dims.push Dimension::new( source.dimension[i].size )
            else
              dims.push Dimension::new
            end
          else
            dims.push Dimension::new( slice.length )
          end
        end
      }
      return dims
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

    def align?
      return !!@alignment
    end

    def set_align(align)
      @alignment = align
      return self
    end

    def to_var
      var = @source.copy("#{self}", :const => nil, :constant => nil, :dim => nil, :dimension => nil, :direction => nil, :dir => nil, :align => alignment)
      return var
    end

    def [](*args)
      slice = false
      args.each { |a|
        slice = true if a.kind_of?(Range) or a.kind_of?(Array) or a.kind_of?(Symbol) or a.nil?
      }
      new_args = []
      slices.each { |s|
        if not s.scalar?
          raise "Invalid slice!" if args.length == 0
          new_arg = SliceItem::new(args.shift)
          new_arg.recurse!(s)
          new_args.push new_arg
        else
          new_args.push s
        end
      }
      if slice then
        return Slice::new(@source, *new_args)
      else
        return Index::new(@source, *new_args)
      end
    end

    private

    def to_s_c
      s = "#{@source}["
      dims = @source.dimension.reverse
      slices = @slices.reverse
      slices_to_c = []
      slices.each_index { |indx|
        slice = slices[indx]
        raise "C does not support slices with step!" if slice.step
        if slice.all? then
          slices_to_c.push(":")
        else
          start = Expression::new(Substraction, slice.first, dims[indx].start)
          if slice.scalar? then
            slices_to_c.push("#{start}")
          else
            slices_to_c.push("#{start}:#{slice.length}")
          end
        end
      }
      return s + slices_to_c.join("][") + "]"
    end

    def to_s_fortran
      slices_to_fortran = @slices.collect { |slice|
        if slice.all? then
          ":"
        else
          s = "#{slice.first}"
          if not slice.scalar? then
            s += ":#{slice.last}"
            s += ":#{slice.step}" if slice.step
          end
          s
        end
      }
      return "#{source}(#{slices_to_fortran.join(",")})"
    end

  end

  class Variable

    def slice(*slices)
      Slice::new(self, *slices)
    end

  end

end
