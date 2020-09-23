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
    attr_reader :rules
    attr_reader :checkers
    HASH_NAME = "options"

    def initialize( *parameters )
      @rules = nil
      @checkers = nil
      if parameters.length == 1 and parameters[0].is_a?(Hash) then
        @parameters = []
        parameters[0].each { |key, value|
          if key == :rules then
            @rules = [value].flatten
            format_rules
          elsif key == :checkers then
            @checkers = [value].flatten
          else
            @parameters.push( OptimizationParameter::new(key, value) )
          end
        }
      else
        @parameters = parameters
      end
      if @checkers then
        @checkers.each { |checker| eval checker }
      end
      if @rules then
        s = <<EOF
  def rules_checker(#{HASH_NAME})
    return ( (#{@rules.join(") and (")}) )
  end
EOF
      else
s = <<EOF
  def rules_checker(#{HASH_NAME})
    return true
  end
EOF
      end
      eval s
    end

    # Add to the parameters of the rules the name of the hash variable
    def format_rules
      regxp = /(?<!#{HASH_NAME}\[):\w+(?!\])/
      @rules.each{|r|
        matches = r.scan(regxp)
        matches = matches.uniq
        matches.each{ |m|
          r.gsub!(/(?<!#{HASH_NAME}\[)#{m}(?!\])/, "#{HASH_NAME}[#{m}]")
        }
      }
    end

    # Remove all points that do not meet ALL the rules.
    def remove_unfeasible (points = [])
      if @rules then
        points.select!{ |pt|
          rules_checker(pt)
        }
      end
      return points
    end

    def to_h
      h = {}
      @parameters.each { |p|
        h[p.name] = p.values
      }
      h[:rules] = @rules if @rules
      h[:checkers] = @checkers if @checkers
      return h
    end
  end

end
