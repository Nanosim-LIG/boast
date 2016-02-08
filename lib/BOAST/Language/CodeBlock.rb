module BOAST

  class CodeBlock < Proc
    include PrivateStateAccessor
    include Inspectable
    extend Functor

    attr_accessor :options

    def initialize(options = {},&block)
      @options = options
      super(&block)
    end

  end

end
