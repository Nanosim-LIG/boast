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
    opt_space = OptimizationSpace::new( :elements_number => [1],
                                        :y_component_number => [1],
                                        :threads_number => [32,64,96,128],
                                        :lws_y => [1,2,4,8,16,32,64,96,128],
                                        :rules => [":lws_y <= :threads_number", ":threads_number % :lws_y == 0", ":y_component_number <= :elements_number", ":elements_number % :y_component_number == 0"] 
                                        )
    opt_space.format_rules
    optimizer = BruteForceOptimizer::new(opt_space, :randomize => false)
    optimizer.points.each{ |o|
      assert(o[:lws_y] <= o[:threads_number], " o[:lws_y] <= o[:threads_number] | #{o}")
      assert(o[:threads_number] % o[:lws_y] == 0, "o[:threads_number] % o[:lws_y] | #{o}") 
      assert(o[:y_component_number] <= o[:elements_number], "o[:y_component_number] <= o[:elements_number] | #{o}")
      assert(o[:elements_number] % o[:y_component_number] == 0, "o[:threads_number] % o[:lws_y] == 0 | #{o}")
    }
  end

end

