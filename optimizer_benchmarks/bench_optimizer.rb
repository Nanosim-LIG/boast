[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'BOAST'
include BOAST
require'optparse'

options = {}
opt_parser = OptionParser.new { |opts|
  opts.banner = "Usage: bench_optimizer.rb parameters.yaml data.yaml [options]"

  opts.on("-lNAME", "--log_file=path_to_file.yaml", "specify the path to store the values explored by the genetic algorithm") { |n|
    options[:log_file] = n
  }  
  
  opts.on("-rNAME", "--result_file=path_to_file.yaml", "specify the path to store the best value of the genetic algorithm") { |n|
    options[:result_file] = n
  }

  opts.on("-h", "--help", "Prints this help") {
    puts opts
    exit
  }
}.parse!

opt_space = OptimizationSpace::new( YAML::load( File::read(ARGV[0]) ) )
oracle = YAML::load( File::read(ARGV[1]) )

optimizer = BruteForceOptimizer::new( opt_space )
puts optimizer.optimize { |opts|
  oracle[opts]
}
puts optimizer.experiments

optimizer = GeneticOptimizer::new( opt_space )
exploration_res = {}
best = optimizer.optimize(:generations_limit => 10, :evolution_types => [Darwinning::EvolutionTypes::MutativeReproduction.new(mutation_rate: 0.10) ]) { |opts|
  exploration_res[opts] = oracle[opts]
  oracle[opts]
} 
puts best
puts optimizer.experiments

if options[:result_file]
  File::open( options[:result_file], "a") { |f|
    f.print YAML::dump({best[0] => best[1]})
  }
end

if options[:log_file]
  File::open( options[:log_file], "w") { |f|
    f.print YAML::dump(exploration_res)
  }
end
