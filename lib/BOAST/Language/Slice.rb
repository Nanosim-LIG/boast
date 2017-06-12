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
          @last = Expression::new(Subtraction, @last, 1) if slice.exclude_end?
          @length = Expression::new(Subtraction, @last, @first)
          @length = @length + 1
        elsif slice.kind_of?(Array) then
          @first = slice [0] if slice.length > 0
          @last = slice [1] if slice.length > 1
          @step = slice [2] if slice.length > 2
          @length = Expression::new(Subtraction, @last, @first)
          @length = @length / @step if @step
          @length = @length + 1
        elsif slice.kind_of?(Symbol) then
          raise "Invalid Slice item: #{slice.inspect}!" if slice != :all
        else
          @first = slice
        end
      end

      def copy_slice!(slice)
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
          @first = Expression::new(Subtraction, @first, get_array_start)
          @first = @first * s.step if s.step
          if s.all? then
            @first = Expression::new(Addition, d.start, @first )
          else
            @first = Expression::new(Addition, s.first, @first )
          end
          unless scalar? then
            if s.step then
              if @step then
                @step = Expression::new(Multiplication, @step, s.step)
              else
                @step = s.step
              end
            end
            @last = @first + (@length-1)*@step
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
      raise "Cannot slice a non array Variable!" unless source.dimension?
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
        unless slice.scalar? then
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
      s << indent
      s << to_s
      s << ";" if [C, CL, CUDA].include?( lang )
      output.puts s
      return self
    end

    def align?
      return @alignment
    end

    def set_align(align)
      @alignment = align
      return self
    end

    def to_var
      var = @source.copy("#{self}", :const => nil, :constant => nil, :dim => nil, :dimension => nil, :direction => nil, :dir => nil, :align => alignment)
      return var
    end

    # Indexes a {Slice}
    # @param [Array{#to_s, Range, [first, last, step], :all, nil}] args one entry for each {SliceItem} of the {Slice}.
    #   * Range: if an index is a Range, the result will be a {Slice}. The Range can be exclusive. The first and last item of the Range will be considered first and last index in the corresponding {SliceItem}.
    #   * [first, last, step]: if an index is an Array, the result will be a {Slice}. The first and last item of the array will be considered first and last index in the corresponding {SliceItem}. If a step is given the range will be iterated by step.
    #   * :all, nil: The whole corresponding {SliceItem} will be used for the slice.
    #   * #to_s: If an index is none of the above it will be considered a scalar index. If all indexes are scalar an {Index} will be returned.
    # @return [Slice, Index]
    def [](*args)
      slice = false
      args.each { |a|
        slice = true if a.kind_of?(Range) or a.kind_of?(Array) or a.kind_of?(Symbol) or a.nil?
      }
      new_args = []
      slices.each_with_index { |s, i|
        unless s.scalar?
          raise "Invalid slice!" if args.length == 0
          new_arg = SliceItem::new(args.shift)
          new_arg.recurse!(s, @source.dimension[i])
          new_args.push new_arg
        else
          new_args.push s
        end
      }
      if slice then
        return Slice::new(@source, *new_args)
      else
        return Index::new(@source, *(new_args.collect(&:first)))
      end
    end

    private

    def to_s_c
      dims = @source.dimension.reverse
      slices_to_c = @slices.reverse.each_with_index.collect { |slice, indx|
        if slice.all? then
          ":"
        else
          start = Expression::new(Subtraction, slice.first, dims[indx].start)
          s = "#{start}"
          unless slice.scalar? then
            s << ":#{slice.length}"
#           s << ":#{slice.step}" if slice.step
            raise "Slice don't support step in C!" if slice.step
          end
          s
        end
      }
      return "#{@source}[#{slices_to_c.join("][")}]"
    end

    def to_s_fortran
      slices_to_fortran = @slices.collect { |slice|
        if slice.all? then
          ":"
        else
          s = "#{slice.first}"
          unless slice.scalar? then
            s << ":#{slice.last}"
            s << ":#{slice.step}" if slice.step
          end
          s
        end
      }
      return "#{@source}(#{slices_to_fortran.join(", ")})"
    end

  end

  class Variable

    def slice(*slices)
      Slice::new(self, *slices)
    end

  end

end
