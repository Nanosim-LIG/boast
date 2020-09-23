module BOAST

  class BruteForceOptimizer < Optimizer

    def initialize(search_space, options = {} )
      super
      @randomize = options[:randomize]
      @checkpoint = options[:checkpoint]
      @checkpoint_size = options[:checkpoint_size]
      @seed = options[:seed]
    end

    def to_a
      return each.to_a
    end

    alias points to_a

    def each
      array = @search_space.parameters.collect { |p| [p.name,p.values.to_a] }
      lazy_block = lambda { |rank, data|
        array[rank][1].each { |d|
          data[array[rank][0]] = d
          if rank == array.length - 1 then
            yield data.dup if @search_space.rules_checker(data)
          else
            lazy_block.call(rank+1, data)
          end
        }
      }
      if block_given? then
        lazy_block.call(0, {})
        return self
      else
        return to_enum(:each)
      end
    end

    def each_random( &block)
      self.points.shuffle.each(&block)
      return self if block_given?
    end

    def optimize(&block)
      @experiments = 0
      @log = {}
      best = [nil, Float::INFINITY]
      e = each

      if @randomize then
        e = e.to_a
        (@seed ? e.shuffle!(random: Random.new(@seed)) : e.shuffle!)
      end
      e = e.drop(@checkpoint).take(@checkpoint_size) if @checkpoint_size

      e.each { |config|
        @experiments += 1
        metric = block.call(config)
        @log[config] = metric if optimizer_log
        best = [config, metric] if metric < best[1]
      }
      if optimizer_log_file then
        File::open(File::basename(optimizer_log_file,".yaml")+".yaml", "w") { |f|
          f.print YAML::dump(@log)
        }
        File::open(File::basename(optimizer_log_file,".yaml")+"_parameters.yaml", "w") { |f|
          f.print YAML::dump(@search_space.to_h)
        }
      end
      return best
    end

  end

end
