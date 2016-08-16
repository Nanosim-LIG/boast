[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'BOAST'
gem 'minitest'
require 'minitest/autorun'
include BOAST

class TestOptimizationSpace < Minitest::Unit::TestCase
  def test_format_rules
    opt_space = OptimizationSpace::new(:rules => [":lws_y <= :threads_number", ":threads_number % :lws_y == 0"])

    r = ["#{OptimizationSpace::HASH_NAME}[:lws_y] <= #{OptimizationSpace::HASH_NAME}[:threads_number]", 
         "#{OptimizationSpace::HASH_NAME}[:threads_number] % #{OptimizationSpace::HASH_NAME}[:lws_y] == 0"]

    assert_equal(r[0], opt_space.rules[0])
    assert_equal(r[1], opt_space.rules[1])
  end

  def test_check_rules
    opt_space = OptimizationSpace::new( :rules => [":lws_y <= :threads_number", ":threads_number % :lws_y == 0", ":y_component_number <= :elements_number", ":elements_number % :y_component_number == 0"], 
                                        :elements_number => [1],
                                        :y_component_number => [1],
                                        :threads_number => [32,64,128,256,512,1024],
                                        :lws_y => [1,2,4,8,16,32,64,128,256,512,1024]
                                        )
    opt_space.format_rules
    opt = []
    count = 0
    for j in 0..2
      for i in 0..2
        for l in 1..4
          for k in 1..8
            opt[count] = {:elements_number=>l, :y_component_number => k, :threads_number => 2**j, :lws_y => 2**i}
            count += 1
          end
        end
      end
    end
    
    opt_space.remove_unfeasible opt
    opt.each{ |o|
      assert(o[:lws_y] <= o[:threads_number], " o[:lws_y] <= o[:threads_number] | #{o}")
      assert(o[:threads_number] % o[:lws_y] == 0, "o[:threads_number] % o[:lws_y] | #{o}") 
      assert(o[:y_component_number] <= o[:elements_number], "o[:y_component_number] <= o[:elements_number] | #{o}")
      assert(o[:elements_number] % o[:y_component_number] == 0, "o[:threads_number] % o[:lws_y] == 0 | #{o}")
    }
  end

  def test_bruteforce_point
    checker = <<EOF 
#    def compute_kernel_size (elements_number, y_component_number, vector_length, temporary_size, load_overlap, threads_number)
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
EOF
    opt_space = OptimizationSpace::new( :elements_number => 1..24,
                                        :y_component_number => 1..6,
                                        :vector_length      => [1,2,4,8,16],
                                        :temporary_size     => [2,4],
                                        :vector_recompute   => [true],
                                        :load_overlap       => [true,false],
                                        :threads_number => [32,64,128,256,512,1024],
                                        :lws_y => [1,2,4,8,16,32,64,128,256,512,1024],
                                        :rules => [":lws_y <= :threads_number", 
                                                   ":threads_number % :lws_y == 0",
                                                   ":elements_number >= :y_component_number",
                                                   ":elements_number % :y_component_number == 0", 
                                                   ":elements_number / :y_component_number <= 4",
                                                   "compute_kernel_size(:elements_number, :y_component_number, :vector_length, :temporary_size, :load_overlap, :threads_number) < compute_kernel_size(6,6,8,2,false,1024)"
                                                  ],
                                        :checkers => checker
                                        )
  
    optimizer = BruteForceOptimizer::new(opt_space, :randomize => false)
    eval checker
    optimizer.points.each{ |o|

      assert(o[:lws_y] <= o[:threads_number], " o[:lws_y] <= o[:threads_number] | #{o}")
      assert(o[:threads_number] % o[:lws_y] == 0, "o[:threads_number] % o[:lws_y] | #{o}") 
      assert(o[:y_component_number] <= o[:elements_number], "o[:y_component_number] <= o[:elements_number] | #{o}")
      assert(o[:elements_number] % o[:y_component_number] == 0, "o[:threads_number] % o[:lws_y] == 0 | #{o}")
      assert(o[:elements_number] % o[:y_component_number] <= 4, "o[:elements_number] / o[:y_component_number] <= 4 | #{o}")
      assert(compute_kernel_size(o[:elements_number], o[:y_component_number], o[:vector_length], o[:temporary_size], o[:load_overlap], o[:threads_number]) < compute_kernel_size(6,6,8,2,false,1024), "Checkers failed")
    }
    assert(optimizer.points.length > 0)
    puts "Number of points generated for the brute force : #{optimizer.points.length}"
  end 

  def test_bruteforce_constraint_save
    checker = <<EOF 
   def compute_kernel_size (elements_number, y_component_number, vector_length, temporary_size, load_overlap, threads_number)
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
EOF
    opt_space = OptimizationSpace::new( :elements_number => 1..2,
                                        :y_component_number => 1..2,
                                        :vector_length      => [1],
                                        :temporary_size     => [2,4],
                                        :vector_recompute   => [true],
                                        :load_overlap       => [true,false],
                                        :threads_number => [1024],
                                        :lws_y => [1],
                                        :rules => [":lws_y <= :threads_number", 
                                                   ":threads_number % :lws_y == 0",
                                                   ":elements_number >= :y_component_number",
                                                   ":elements_number % :y_component_number == 0", 
                                                   ":elements_number / :y_component_number <= 4",
                                                   "compute_kernel_size(:elements_number, :y_component_number, :vector_length, :temporary_size, :load_overlap, :threads_number) < compute_kernel_size(6,6,8,2,false,1024)"
                                                  ],
                                        :checkers => checker
                                        )
  
    optimizer = BruteForceOptimizer::new(opt_space, :randomize => false)
    eval checker
    optimizer.points.each{ |o|

      assert(o[:lws_y] <= o[:threads_number], " o[:lws_y] <= o[:threads_number] | #{o}")
      assert(o[:threads_number] % o[:lws_y] == 0, "o[:threads_number] % o[:lws_y] | #{o}") 
      assert(o[:y_component_number] <= o[:elements_number], "o[:y_component_number] <= o[:elements_number] | #{o}")
      assert(o[:elements_number] % o[:y_component_number] == 0, "o[:threads_number] % o[:lws_y] == 0 | #{o}")
      assert(o[:elements_number] % o[:y_component_number] <= 4, "o[:elements_number] / o[:y_component_number] <= 4 | #{o}")
      assert(compute_kernel_size(o[:elements_number], o[:y_component_number], o[:vector_length], o[:temporary_size], o[:load_overlap], o[:threads_number]) < compute_kernel_size(6,6,8,2,false,1024), "Checkers failed")
    }
    assert(optimizer.points.length > 0)
    puts "Number of points generated for the brute force : #{optimizer.points.length}"

    File::open("/tmp/parameters.yaml", "w") { |f|
      f.print YAML::dump(opt_space.to_h)
    }

    new_yaml = YAML::load( File::read("/tmp/parameters.yaml") )
    checker2 = new_yaml[:checkers]
    assert(checker2 == checker)
  end 


  def test_algo_gen_point
    checker = <<EOF
#     def compute_kernel_size (elements_number, y_component_number, vector_length, temporary_size, load_overlap, threads_number)
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
EOF
    opt_space = OptimizationSpace::new( :elements_number => 1..24,
                                        :y_component_number => 1..6,
                                        :vector_length      => [1,2,4,8,16],
                                        :temporary_size     => [2,4],
                                        :vector_recompute   => [true],
                                        :load_overlap       => [true,false],
                                        :threads_number => [32,64,128,256,512,1024],
                                        :lws_y => [1,2,4,8,16,32,64,128,256,512,1024],
                                        :rules => [":lws_y <= :threads_number", 
                                                   ":threads_number % :lws_y == 0",
                                                   ":elements_number >= :y_component_number",
                                                   ":elements_number % :y_component_number == 0", 
                                                   ":elements_number / :y_component_number <= 4",
                                                   "compute_kernel_size(:elements_number, :y_component_number, :vector_length, :temporary_size, :load_overlap, :threads_number) < compute_kernel_size(6,6,8,2,false,1024)"
                                                  ],
                                        :checkers => checker 
                                        )

    optimizer = GeneticOptimizer::new(opt_space)
    optimizer.optimize(:generations_limit => 10, :evolution_types => [Darwinning::EvolutionTypes::MutativeReproduction.new(mutation_rate: 0.00) ]){ |opts|
      rand(100)
    }

    eval checker
    optimizer.history.each{|h|
      h.each{|g|
          assert(g.to_a[0][:lws_y] <= g.to_a[0][:threads_number], "#{g.to_a[0][:lws_y]} <= #{g.to_a[0][:threads_number]}")
          assert(g.to_a[0][:threads_number] % g.to_a[0][:lws_y] == 0, "#{g.to_a[0][:threads_number]} % #{g.to_a[0][:lws_y]} = #{g.to_a[0][:threads_number] % g.to_a[0][:lws_y]}") 
          assert(g.to_a[0][:elements_number] >= g.to_a[0][:y_component_number], "#{g.to_a[0][:elements_number]} >= #{g.to_a[0][:y_component_number]}")
          assert(g.to_a[0][:elements_number] % g.to_a[0][:y_component_number] == 0, "#{g.to_a[0][:elements_number]} % #{g.to_a[0][:y_component_number]} == #{g.to_a[0][:elements_number] % g.to_a[0][:y_component_number]}")
          assert(g.to_a[0][:elements_number] / g.to_a[0][:y_component_number] <= 4, "elements_number / y_component_number <= 4 | #{g.to_a[0][:elements_number] / g.to_a[0][:y_component_number]}")
          assert(compute_kernel_size(g.to_a[0][:elements_number], g.to_a[0][:y_component_number], g.to_a[0][:vector_length], g.to_a[0][:temporary_size], g.to_a[0][:load_overlap], g.to_a[0][:threads_number]) < compute_kernel_size(6,6,8,2,false,1024), "Checkers failed")
      }
    }
    assert(optimizer.history.flatten(1).length > 0)
    puts "Number of points for genetic algorithm : #{optimizer.history.flatten(1).length}"
  end

end

