module BOAST

  class Optimizer
    include PrivateStateAccessor
    attr_reader :experiments
    attr_reader :search_space
    attr_reader :log
    attr_reader :history

    def initialize(search_space, options = {} )
      @search_space = search_space
      @experiments = 0
      @log = {}
    end
  end

end

