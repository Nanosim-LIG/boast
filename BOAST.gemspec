Gem::Specification.new do |s|
  s.name = 'BOAST'
  s.version = "2.0.1"
  s.author = "Brice Videau"
  s.email = "brice.videau@imag.fr"
  s.homepage = "https://github.com/Nanosim-LIG/boast"
  s.summary = "BOAST is a computing kernel metaprogramming tool."
  s.description = "BOAST aims at providing a framework to metaprogram, benchmark and validate computing kernels"
  s.files = Dir['BOAST.gemspec', 'LICENSE', 'README.md', 'lib/**/*']
  s.has_rdoc = false
  s.license = 'BSD-2-Clause'
  s.required_ruby_version = '>= 1.9.3'
  s.add_dependency 'narray', '~> 0.6.0', '>=0.6.0.8'
  s.add_dependency 'narray_ffi', '~> 1.2', '>=1.2.0'
  s.add_dependency 'opencl_ruby_ffi', '~> 1.3', '>=1.3.2'
  s.add_dependency 'systemu', '~> 2', '>=2.2.0'
  s.add_dependency 'os', '~> 0.9', '>=0.9.6'
  s.add_dependency 'PAPI', '~> 0', '>=0.101'
  s.add_dependency 'ffi', '~> 1.9', '>=1.9.3'
  s.add_dependency 'rgl', '~> 0.5', '>=0.5.1'
  s.add_dependency 'rake', '>=0.9'
end
