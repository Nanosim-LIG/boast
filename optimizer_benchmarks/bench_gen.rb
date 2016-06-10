[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'BOAST'
include BOAST
require'optparse'

options = {}
opt_parser = OptionParser.new { |opts|
  opts.banner = "Usage: bench_optimizer.rb parameters.yaml data.yaml [options]"

  opts.on("-eVAL", "--elitism=VAL", Integer, "Specify the elitism value for the genetic algorithm") { |n|
    options[:elitism] = n
  }  

  opts.on("-gLIM", "--generations_limit=RATE", Integer,"Specify the generation limit for the genetic algorithm") { |n|
    options[:generations_limit] = n
  }  

  opts.on("-mRATE", "--mutation_rate=RATE", Float,"Specify the mutation rate for the genetic algorithm") { |n|
    options[:mutation_rate] = n
  }  

  opts.on("-pSIZE", "--population_size=SIZE", Integer,"Specify the population size for the genetic algorithm") { |n|
    options[:population_size] = n
  }  

  opts.on("-tBOOL", "--twin_removal=BOOL", "Specify the twin removal value for the genetic algorithm") { |n|
    options[:twin_removal] = n == "true" ? true : false
  }  

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

optimizer = GeneticOptimizer::new( opt_space )

exploration_logs = {}
logs = {}

res = []
# logs[gen_opt] = []
(1..1000).each{
  best = optimizer.optimize(:generations_limit => options.fetch(:generations_limit,10), 
                            :evolution_types => [Darwinning::EvolutionTypes::MutativeReproduction.new(mutation_rate: options.fetch(:mutation_rate,0.10))],
                            :population_size => options.fetch(:population_size, 10),
                            :elitism => options.fetch(:elitism,1),
                            :twin_removal => options.fetch(:twin_removal,true)
                            ) { |opts|
    exploration_logs[opts] = oracle[opts]
    oracle[opts]
  }
  res.push best
  # logs[gen_opt].push best[1]
}

puts options[:result_file]

if options[:result_file]
  File::open( options[:result_file], "w") { |f|
    f.print YAML::dump(res)
  }
end

if options[:log_file]
  File::open( options[:log_file], "w") { |f|
    f.print YAML::dump(exploration_logs)
  }
end
