Gem::Specification.new do |s|
  s.name = 'BOAST'
  s.version = "0.3"
  s.author = "Brice Videau"
  s.email = "brice.videau@imag.fr"
  s.homepage = "https://forge.imag.fr/projects/boast/"
  s.summary = "BOAST is a computing kernel metaprogramming tool."
  s.description = "BOAST aims at providing a framework to metaprogram, benchmark and validate computing kernels"
  s.files = %w( BOAST.gemspec LICENSE lib/BOAST.rb lib/BOAST/Algorithm.rb lib/BOAST/CKernel.rb lib/BOAST/BOAST_OpenCL.rb lib/BOAST/Transitions.rb)
  s.has_rdoc = true
  s.license = 'BSD'
  s.required_ruby_version = '>= 1.9.3'
  s.add_dependency 'narray', '>=0.6.0.8'
  s.add_dependency 'opencl_ruby_ffi', '>=0.4'
  s.add_dependency 'systemu', '>=2.2.0'
end
