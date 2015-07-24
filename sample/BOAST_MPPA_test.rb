# -*- coding: utf-8 -*-
[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'rubygems'
require 'narray'
require 'BOAST'

io_code = <<EOF
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
int hello(int32_t a, int32_t* b){
  int pid, ret, to_clust;
  printf("Hello from IO Cluster\\n");
  
  to_clust = mppa_open("/mppa/rqueue/0:10/128:10/1.4", O_WRONLY);
  assert(to_clust != -1);

  printf("IO : Received value = %d\\n", a);

  ret =  mppa_write(to_clust, &a, sizeof(a));
  assert(ret != -1);
  
  *b = a + 1;

  ret = mppa_close(to_clust);
  assert(ret != -1);

  return 0;
}
EOF

comp_code = <<EOF
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
int hello(){
  int from_io, ret;
  uint32_t buf;
  
  from_io = mppa_open("/mppa/rqueue/0:10/128:10/1.4", O_RDONLY);
  assert(from_io != -1);

  printf("Hello from Compute Cluster\\n");
  
  ret = mppa_read(from_io, &buf, sizeof(buf));
  assert(ret != -1);

  printf("Cluster : Received value = %d\\n", buf);

  ret = mppa_close(from_io);
  assert(ret != -1);

  return 0;
}
EOF

BOAST::set_architecture(BOAST::MPPA)
BOAST::set_lang(BOAST::C)

kernel = BOAST::CKernel::new
BOAST::get_output.write comp_code
kernel.set_io
BOAST::get_output.write io_code
kernel.set_comp
a = BOAST::Int("a", :dir=>:in)
b = BOAST::Int("b", :dir=>:out)
kernel.procedure = BOAST::Procedure("hello", [a, b])
kernel.build
r = kernel.run(42, 0, :clust_list => [0])

puts "BOAST : Received value =  #{r[:reference_return][:b]}"
puts "BOAST : Average power = #{r[:avg_pwr].round(3)} W"
puts "BOAST : Energy consumption = #{r[:energy].round(3)} J"
puts "BOAST : Raw results = #{r.to_s}"
sleep 2
