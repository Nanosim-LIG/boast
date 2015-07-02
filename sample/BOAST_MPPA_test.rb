# -*- coding: utf-8 -*-
[ '../lib', 'lib' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'rubygems'
require 'narray'
require 'BOAST'

io_code = <<EOF
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
int hello(){
  int pid, ret, from_host, to_clust, to_host;
  int32_t buf;
  printf("Hello from IO Cluster\\n");
  
  pid = mppa_spawn(0, NULL, "comp-part", NULL, NULL);
  assert(pid != -1);

  from_host = mppa_open("/mppa/buffer/board0#mppa0#pcie0#3/host#3", O_RDONLY);
  assert(from_host != -1);

  to_clust = mppa_open("/mppa/rqueue/0:10/128:10/1.4", O_WRONLY);
  assert(to_clust != -1);

  ret = mppa_read(from_host, &buf, sizeof(buf));
  assert(ret != -1);

  // buf=666;
  printf("IO : Received value = %d\\n", buf);

  ret =  mppa_write(to_clust, &buf, sizeof(buf));
  assert(ret != -1);

  to_host = mppa_open("/mppa/buffer/host#5/board0#mppa0#pcie0#5", O_WRONLY);
  assert(to_host != -1);
  
  buf++;
  ret = mppa_write(to_host, &buf, sizeof(buf));  
  assert(ret != -1);

  ret = mppa_close(to_clust);
  assert(ret != -1);

  ret = mppa_close(from_host);
  assert(ret != -1);

  ret = mppa_close(to_host);
  assert(ret != -1);

  ret = mppa_waitpid(pid, NULL, 0);
  assert(ret != -1);

  mppa_exit(0);
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

  mppa_exit(0);
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
r = kernel.run(66, 0)

puts "BOAST : Valeur retournée #{r[:reference_return][:b]}"

sleep 2
