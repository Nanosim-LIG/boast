module BOAST

  # Generates setters and getters for the specified state
  # @param [Symbol] state
  # @!macro [attach] state_accessor
  #   @!method $1
  #     @scope class
  #     @return the BOAST *$1* state
  #   @!method $1=( val )
  #     @scope class
  #     Sets *$1* state to a new value
  #     @param val the new value of *$1* state
  #     @return the new +$1+ state
  #   @!method get_$1
  #     @scope class
  #     @return the *$1* state
  #   @!method set_$1( val )
  #     @scope class
  #     Sets *$1* state to a new value
  #     @param val the new value of *$1* state
  #     @return the new *$1* state
  def self.state_accessor(state)
    s = <<EOF
  module_function

  def #{state}=(val)
    @@#{state} = val
  end

  def #{state}
    @@#{state}
  end

  def set_#{state}(val)
    @@#{state} = val
  end

  def get_#{state}
    @@#{state}
  end
EOF
    eval s
  end

  # Generates setters and getters for the specified boolean state
  # @param [Symbol] state
  # @!macro [attach] boolean_state_accessor
  #   @!method $1?
  #     @scope class
  #     @return the boolean evaluation of the *$1* state
  #   @!parse state_accessor $1
  def self.boolean_state_accessor(state)
    state_accessor(state)
    s = <<EOF
  module_function

  def #{state}?
    !!@@#{state}
  end
EOF
    eval s
  end


  # Generates an initializer for the specified state using default value or environment variable. Calls this initializer.
  # @param [Symbol] state
  # @param [Object] default default value
  # @param [String] get_env_string if specified, an escaped string that can be evaluated. the envs variable can be used in the string to obtain what the corresponding environment variable was. Example: '"const_get(#{ envs })"'
  # @param [Symbol] env name of the corresponding environment variable
  # @!macro [attach] default_state_getter
  #   @!method get_default_$1
  #     @scope class
  #     @private
  def self.default_state_getter(state, default, get_env_string=nil, env = state.upcase)
    envs = "ENV['#{env}']"
    s = <<EOF
  module_function

  def get_default_#{state}
    #{state} = @@boast_config[#{state.inspect}]
    #{state} = #{default.inspect} unless #{state}
    #{state} = #{get_env_string ? eval( "#{get_env_string}" ) : "YAML::load(#{envs})" } if #{envs}
    return #{state}
  end

  @@#{state} = get_default_#{state}
EOF
    eval s
  end

  # Implements private setters and getters interface for BOAST states.
  module PrivateStateAccessor

    # Generates private setters and getters for the specified state
    # @param [Symbol] state
    # @!macro [attach] private_state_accessor
    #   @!method $1
    #     @return the BOAST *$1* state
    #   @!method $1=( val )
    #     Sets *$1* state to a new value
    #     @param val the new value of *$1* state
    #     @return the new +$1+ state
    #   @!method get_$1
    #     @return the *$1* state
    #   @!method set_$1( val )
    #     Sets *$1* state to a new value
    #     @param val the new value of *$1* state
    #     @return the new *$1* state
    def self.private_state_accessor(state)
      s = <<EOF
    private
    def #{state}=(val)
      BOAST::set_#{state}(val)
    end
    def #{state}
      BOAST::get_#{state}
    end
    def set_#{state}(val)
      BOAST::set_#{state}(val)
    end
    def get_#{state}
      BOAST::get_#{state}
    end
EOF
      eval s
    end
  
    # Generates private setters and getters for the specified boolean state
    # @param [Symbol] state
    # @!macro [attach] private_boolean_state_accessor
    #   @!method $1?
    #     @return the boolean evaluation of the *$1* state
    #   @!parse private_state_accessor $1
    def self.private_boolean_state_accessor(state)
      self.private_state_accessor(state)
      s = <<EOF
    private
    def #{state}?
      BOAST::#{state}?
    end
EOF
      eval s
    end

  end

end

