[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'BOAST'
gem 'minitest'
require 'minitest/autorun'
include BOAST

class TestOptimizationSpace < Minitest::Test

  def test_format_rules
    opt_space = OptimizationSpace::new(:rules => [":lws_y <= :threads_number", ":threads_number % :lws_y == 0"])

    r = ["#{OptimizationSpace::HASH_NAME}[:lws_y] <= #{OptimizationSpace::HASH_NAME}[:threads_number]",
         "#{OptimizationSpace::HASH_NAME}[:threads_number] % #{OptimizationSpace::HASH_NAME}[:lws_y] == 0"]

    assert_equal(r[0], opt_space.rules[0])
    assert_equal(r[1], opt_space.rules[1])
  end

  def test_optimization_space_to_enum
    opt_space = OptimizationSpace::new( :param => 1..30, :opt => [1,3] )
    optimizer = BruteForceOptimizer::new( opt_space )
    points = optimizer.each.to_a
    assert_equal((1..30).size*[1,3].length, points.length)
    assert_equal(points.length, points.uniq.length)
  end

  def test_optimization_space
    opt_space = OptimizationSpace::new( :param => 1..30, :opt => [1,3] )
    optimizer = BruteForceOptimizer::new( opt_space )
    points = optimizer.points
    assert_equal((1..30).size*[1,3].length, points.length)
    assert_equal(points.length, points.uniq.length)
  end

  def test_check_rules_to_enum
    opt_space = OptimizationSpace::new( :param => 1..30,
                                        :rules => [":param % 2 != 0",
                                                   ":param % 7 != 0"] )
    optimizer = BruteForceOptimizer::new( opt_space )
    count = (1..30).to_a.reject! { |e| e % 2 == 0 or e % 7 == 0 }.length
    points = optimizer.each.to_a
    assert_equal(count, points.length)
    assert_nil(points.find { |e| e[:param] % 2 == 0 or e[:param] % 7 == 0 })
  end

  def test_check_rules
    opt_space = OptimizationSpace::new( :param => 1..30,
                                        :rules => [":param % 2 != 0",
                                                   ":param % 7 != 0"] )
    optimizer = BruteForceOptimizer::new( opt_space )
    count = (1..30).to_a.reject! { |e| e % 2 == 0 or e % 7 == 0 }.length
    points = optimizer.points
    assert_equal(count, points.length)
    assert_nil(points.find { |e| e[:param] % 2 == 0 or e[:param] % 7 == 0 })
  end

  def test_checker
    check_param_code = <<EOF
    def check_param( p )
      return (p % 2 != 0 and p % 7 != 0 )
    end
EOF
    opt_space = OptimizationSpace::new( :param => 1..30,
                                        :rules => ["check_param(:param)"],
                                        :checkers => check_param_code )
    optimizer = BruteForceOptimizer::new( opt_space )
    count = (1..30).to_a.reject! { |e| e % 2 == 0 or e % 7 == 0 }.length
    points = optimizer.points
    assert_equal(count, points.length)
    assert_nil(points.find { |e| e[:param] % 2 == 0 or e[:param] % 7 == 0 })
  end

  def test_optimization_space_save
    check_param_code = <<EOF
    def check_param( p )
      return (p % 2 != 0 and p % 7 != 0 )
    end
EOF
    opt_space1 = OptimizationSpace::new( :param => 1..30,
                                         :rules => ["check_param(:param)"],
                                         :checkers => check_param_code )
    h = opt_space1.to_h
    opt_space2 = OptimizationSpace::new( h )
    assert_equal(BruteForceOptimizer::new( opt_space1 ).points,
                 BruteForceOptimizer::new( opt_space2 ).points)
  end

  def test_algo_gen
    check_param_code = <<EOF
    def check_param( p )
      return (p % 2 != 0 and p % 7 != 0 )
    end
EOF
    opt_space = OptimizationSpace::new( :param => 1..30,
                                        :rules => ["check_param(:param)"],
                                        :checkers => check_param_code )
    optimizer = GeneticOptimizer::new( opt_space )
    optimizer.optimize( :generations_limit => 10, :twin_removal => false, :evolution_types => [Darwinning::EvolutionTypes::MutativeReproduction.new(mutation_rate: 0.00)] ) { |opts|
      rand
    }
    points = optimizer.history.flatten(1).collect { |ind| ind.to_a[0] }
    assert_nil(points.find { |e| e[:param] % 2 == 0 or e[:param] % 7 == 0 })
    assert_equal(220, points.length)
  end

end

