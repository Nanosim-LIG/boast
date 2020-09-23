require 'cconfigspace'

module BOAST
  class CCSOptimizer < Optimizer
    attr_reader :ccs_tuner
    attr_reader :budget
    def initialize(ccs_tuner_klass, search_space, budget: nil, seed: nil, name: "")
      cs = CCS::ConfigurationSpace::new(name: "#{name}cs")
      search_space.parameters.each { |p|
        n = p.name
        values = p.values
        n = n.inspect if n.kind_of? Symbol
        n = n.to_s
        if values.kind_of? Range
          h = CCS::DiscreteHyperparameter::new(name: n.to_s, values: values.to_a)
        else
          vs = values.to_a
          if vs.all? { |e| e.kind_of?(Numeric) }
            h = CCS::DiscreteHyperparameter::new(name: n.to_s, values: vs.sort)
          else
            h = CCS::CategoricalHyperparameter::new(name: n.to_s, values: vs)
          end
        end
        cs.add_hyperparameter(h)
      }
      os = CCS::ObjectiveSpace::new(name: "#{name}os")
      h = CCS::NumericalHyperparameter::new(name: "#{name}objective")
      e = CCS::Variable::new(hyperparameter: h)
      os.add_hyperparameter(h)
      os.add_objective(e)
      t = ccs_tuner_klass::new(configuration_space: cs, objective_space: os, name: name.to_s)
      @ccs_tuner = t
      @budget = budget
      @seed = seed
      @search_space = search_space
    end

    def optimize(&block)
      t = @ccs_tuner
      t.rng.seed = @seed if @seed
      return nil if @budget && budget < 1
      count = 0
      os = t.objective_space
      while (config = t.ask.first)
        conf = config.to_h
        conf = conf.transform_keys { |k| k.match(/^:/) ? eval(k) : k }
        res = block.call(conf)
        e = CCS::Evaluation::new(objective_space: os, configuration: config, values: [res])
        t.tell([e])
        count += 1
        break if @budget && count>= @budget
      end
      best = t.optimums.first
      conf = best.configuration.to_h
      conf = conf.transform_keys { |k| k.match(/^:/) ? eval(k) : k }
      [conf, best.values.first]
    end

    def experiments
      @ccs_tuner.history_size
    end

    def log
      @ccs_tuner.history.collect { |e| [e.configuration.to_h.transform_keys { |k| k.match(/^:/) ? eval(k) : k }, e.values.first] }
    end
  end
end
