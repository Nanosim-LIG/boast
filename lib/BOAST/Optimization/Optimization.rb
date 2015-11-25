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

  class OptimizationSpace
    attr_reader :parameters

    def initialize( *parameters )
      if parameters.length == 1 and parameters[0].is_a?(Hash) then
        @parameters = []
        parameters[0].each { |key, value|
          @parameters.push( OptimizationParameter::new(key, value) )
        }
      else
        @parameters = parameters
      end
    end

  end

  class BruteForceOptimizer
    def initialize(search_space, options = {} )
      @search_space = search_space
      @randomize = options[:randomize]
    end

    def points
      params2 = @search_space.parameters.dup
      param = params2.shift
      pts = param.values.collect { |val| {param.name => val} }
      if params2.size == 0 then
        return pts
      else
        optim2 = BruteForceOptimizer::new(OptimizationSpace::new(*params2))
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

    def optimize(&block)
      best = [nil, Float::INFINITY]
      pts = points
      pts.shuffle! if @randomize
      enumerator = pts.each { |config|
        metric = block.call(config)
        best = [config, metric] if metric < best[1]
      }
      return best
    end

  end

  class GenericOptimization

    attr_accessor :repeat
    attr_reader :parameters

    def size
      return @parameters.size
    end

    def points
      params2 = @parameters.dup
      param = params2.shift
      pts = param.values.collect { |val| {param.name => val} }
      if params2.size == 0 then
        return pts
      else
        optim2 = GenericOptimization::new(*params2)
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
