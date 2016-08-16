[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'minitest/autorun'
require 'BOAST'
include BOAST
require_relative 'Energy/EnergyProbe'
require_relative 'helper'
