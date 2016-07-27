module BOAST
if false
  module OpenCLRuntime

    st = BOAST::get_output

    p = BOAST::Procedure("energy_kernel_init",[])
    ENERGY_PROBE_INIT = BOAST::CKernel::new( :lang => C )
    ENERGY_PROBE_INIT.procedure = p
    BOAST::get_output.print <<EOF
#include <redfst.h>
void energy_kernel_init(void) {
  redfst_init();
}
EOF

    p = BOAST::Procedure("energy_kernel_start",[])
    ENERGY_PROBE_START = BOAST::CKernel::new( :lang => C )
    ENERGY_PROBE_START.procedure = p
    BOAST::get_output.print <<EOF
#include <redfst.h>
void energy_kernel_start(void) {
  redfst_reset();
}
EOF

    p = BOAST::Procedure("energy_reading_kernel",[BOAST::Real("data",:dim => BOAST::Dim(),:dir => :out)], [], :return => BOAST::Int("nb_cpus") )
    ENERGY_PROBE_STOP = BOAST::CKernel::new( :lang => C )
    ENERGY_PROBE_STOP.procedure = p
    BOAST::get_output.print <<EOF
#include <redfst.h>
int energy_reading_kernel( double * data) {
  redfst_get_all(data);
  return redfst_ncpus();
}
EOF

    ENERGY_PROBE_INIT.build( :LDFLAGS => '-lredfst -lm -lcpufreq' )
    ENERGY_PROBE_START.build( :LDFLAGS => '-lredfst -lm -lcpufreq' )
    ENERGY_PROBE_STOP.build( :LDFLAGS => '-lredfst -lm -lcpufreq' )

    BOAST::set_output(st)

  end
end
end
