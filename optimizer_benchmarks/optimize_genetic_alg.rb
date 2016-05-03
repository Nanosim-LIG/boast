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

# Compute the size of the kernel
checker = Proc.new do |elements_number, y_component_number, vector_length, temporary_size, load_overlap, threads_number|
  vector_number = ((elements_number / y_component_number).to_f / vector_length).ceil
  l_o = load_overlap ? 1 : 0
  
  tempload = (1 - l_o) * (vector_number * vector_length) / vector_length * vector_length
  temp =  l_o * 3 * vector_number * (y_component_number+2) * vector_length
  res = vector_number * y_component_number * vector_length
  tempc = 3 * vector_number * (y_component_number + 2) * temporary_size * vector_length
  out_vec = (1 - l_o) * tempc
  resc = vector_number * y_component_number * temporary_size * vector_length
  
  (tempload + temp + res + tempc + out_vec + resc) * threads_number
end

opt_space = OptimizationSpace::new( YAML::load( File::read(ARGV[0]) ) )
oracle = YAML::load( File::read(ARGV[1]) )

opt_space.checkers.push checker

gen_space = OptimizationSpace::new( :generations_limit => [1,2,4,10,20,25,50,100],
                                    :population_size   => [1,2,4,10,20,25,50,100],
                                    :mutation_rate     => [0.0,0.1, 0.25, 0.50, 0.75, 1.0],
                                    :elitism           => [1,5,10,20],
                                    :twin_removal      => [true,false],
                                    :rules             => [":generations_limit * :population_size == 100"]
                                    )

optimizer = GeneticOptimizer::new( opt_space )
optimizer_optimizer = BruteForceOptimizer::new( gen_space )

exploration_logs = {}
logs = {}

winner = optimizer_optimizer.optimize do |gen_opt|
  res = []
  logs[gen_opt] = []
  (1..30).each{
    best = optimizer.optimize(:generations_limit => gen_opt[:generations_limit], 
                              :evolution_types => [Darwinning::EvolutionTypes::MutativeReproduction.new(mutation_rate: gen_opt[:mutation_rate])],
                              :population_size => gen_opt[:population_size],
                              :elitism => gen_opt[:elitism],
                              :twin_removal => gen_opt[:twin_removal]
                              ) { |opts|
      exploration_logs[opts] = oracle[opts]
      oracle[opts]
    }
    res.push best[1]
    logs[gen_opt].push best[1]
  }
  res.reduce(:+) / res.length
end

puts winner

if options[:result_file]
  File::open( options[:result_file], "w") { |f|
    f.print YAML::dump(logs)
  }
end

if options[:log_file]
  File::open( options[:log_file], "w") { |f|
    f.print YAML::dump(exploration_logs)
  }
end
