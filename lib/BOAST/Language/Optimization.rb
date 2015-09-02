module BOAST

  class OptimizationParameter
    attr_reader :name
    attr_reader :values
    def initialize( name, values )
      @name = name
      @values = values
    end
  end

  class BooleanParameter < OptimizationParameter
    def initialize( name )
      super( name, [false, true] )
    end
  end

  OP = OptimizationParameter
  BP = BooleanParameter
 
  class GenericOptimization

    attr_accessor :repeat
    attr_reader :parameters
  
 
    def size
      return @parameters.size
    end
 
    def points
      pts=[]
      params2 = @parameters.dup
      param = params2.shift
      optim2 = GenericOptimization::new(*params2)
      param.values.each{ |val| 
        pts.push({param.name.to_sym => val})
      }
      if optim2.size == 0 then
        return pts
      else
        pts3=[]
        pts.each{ |p1| 
          optim2.each { |p2| 
            pts3.push(p1.dup.update(p2))
          }
        }
        return pts3
      end
    end

    def each(&block)
      return self.points.each(&block)
    end

    def each_random(&block)
      return self.points.shuffle.each(&block)
    end

    def initialize( *parameters )
      if parameters.length == 1 and parameters[0].is_a?(Hash) then
        @parameters = []
        parameters[0].each { |key, value|
          @parameters.push( OptimizationParameter::new(key, value) )
        }
      else
        @parameters = parameters
      end
      @repeat = 3
    end

  end

end
