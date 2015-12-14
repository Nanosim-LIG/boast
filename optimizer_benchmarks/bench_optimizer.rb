[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'BOAST'
include BOAST

opt_space = OptimizationSpace::new( YAML::load( File::read(ARGV[0]) ) )
oracle = YAML::load( File::read(ARGV[1]) )

optimizer = BruteForceOptimizer::new( opt_space )
puts optimizer.optimize { |opts|
  oracle[opts]*1000000000
}
puts optimizer.experiments

optimizer = GeneticOptimizer::new( opt_space )
puts optimizer.optimize(:generations_limit => 10, :evolution_types => [Darwinning::EvolutionTypes::MutativeReproduction.new(mutation_rate: 0.10) ]) { |opts|
  oracle[opts]*1000000000
} 
puts optimizer.experiments
