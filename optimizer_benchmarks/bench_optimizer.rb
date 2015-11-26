[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'BOAST'
include BOAST

optimizer = BruteForceOptimizer::new( OptimizationSpace::new( YAML::load( File::read(ARGV[0]) ) ) )
oracle = YAML::load( File::read(ARGV[1]) )
puts optimizer.optimize { |opts|
  oracle[opts]
}
puts optimizer.experiments
