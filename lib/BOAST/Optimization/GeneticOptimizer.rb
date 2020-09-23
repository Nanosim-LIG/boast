module BOAST

  class GeneticOptimizer < Optimizer

    def initialize(search_space, options = {} )
      super
      require 'darwinning'
      s = <<EOF
      @organism = Class::new(Darwinning::Organism) do
        @@block = nil
        def self.block
          return @@block
        end
        def self.block=(block)
          @@block = block
        end
        @@experiments = 0
        def self.experiments
          return @@experiments
        end
        def self.experiments=(experiments)
          @@experiments = experiments
        end
EOF
      @search_space.parameters.each { |param|
      s += <<EOF
        @genes.push( Darwinning::Gene.new(name: #{param.name.inspect}, value_range: #{param.values.inspect}) )
EOF
      }
      s += <<EOF
        def initialize(*args)
          super
        end

        def fitness
          return @fitness if @fitness
          opts = {}
          genes.each { |gene|
            opts[gene.name] = genotypes[gene]
          }
          @fitness = @@block.call(opts)
          @@experiments = @@experiments + 1
          return fitness
        end

        def to_a
          opts = {}
          genes.each { |gene|
            opts[gene.name] = genotypes[gene]
          }
          return [opts, fitness]
        end

        def to_s
          return to_a.to_s
        end

      end
EOF
      eval s
    end

    def optimize(options={}, &block)
      opts = { :population_size => 20,
               :fitness_goal => 0,
               :generations_limit => 100,
               :search_space => @search_space }
      opts.update(options)
      opts[:organism] = @organism
      @organism.block = block
      @organism.experiments = 0
      population = Darwinning::Population.new(opts)
      population.evolve!
      @history = population.history
      @experiments = @organism.experiments
      return population.best_member.to_a
    end

  end

end
