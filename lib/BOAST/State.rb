module BOAST

  def self.state_accessor(*args)
    args.each { |arg|
      s = <<EOF
  def #{arg}=(val)
    @@#{arg} = val
  end
  module_function :#{arg}=
  def #{arg}
    @@#{arg}
  end
  module_function :#{arg}
  def set_#{arg}(val)
    @@#{arg} = val
  end
  module_function :set_#{arg}
  def get_#{arg}
    @@#{arg}
  end
  module_function :get_#{arg}
EOF
      eval s
    }
  end

  def self.boolean_state_accessor(*args)
    self.state_accessor(*args)
    args.each { |arg|
      s = <<EOF
  def #{arg}?
    !!@@#{arg}
  end
  module_function :#{arg}?
EOF
      eval s
    }
  end

  module PrivateStateAccessor

    def self.private_state_accessor(*args)
      args.each { |arg|
        s = <<EOF
    private
    def #{arg}=(val)
      BOAST::#{arg}= val
    end
    def #{arg}
      BOAST::#{arg}
    end
    def set_#{arg}(val)
      BOAST::set_#{arg}(val)
    end
    def get_#{arg}
      BOAST::get_#{arg}
    end
EOF
        eval s
      }
    end
  
    def self.private_boolean_state_accessor(*args)
      self.private_state_accessor(*args)
      args.each { |arg|
        s = <<EOF
    private
    def #{arg}?
      BOAST::#{arg}?
    end
EOF
        eval s
      }
    end

  end

end

