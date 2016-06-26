module BOAST
  module Energy
    class Redfst
#      require 'inline'
#      inline(:C) do |b|
#        b.add_compile_flags '-lredfst'
#        b.include '"redfst.h"'
#        b.c 'void energy_init(){}'
#      end
      def initialize
#        energy_init
        true
      end
      def start
#        self.asdqwe
      end
      def stop
      end
      def compute
      end
    end
    @@impl = Redfst.new
    module_function
    def config_validate
    end
    def config_read
    end
    def get
    end
    def initialize
    end
    def start
#      puts @@impl
#     @impl.start
    end
    def stop
#     impl.stop
    end
    def compute
    end
  end

#  module EnergyRedfstProbe
#    extend PrivateStateAccessor
#    module_function
#    def header
#      get_output.print "#include <redfst.h>\n"
#    end
#    def decl
#      get_output.print "double *energyData;\n"
#    end
#    def configure
#      get_output.print '
#//      rb_hash_aset(_boast_stats,ID2SYM(rb_intern("energy")), rb_float_new(1.0));
#      redfst_init();
#      energyData = malloc((3 * redfst_ncpus() + 1) * sizeof(*energyData));
#      '
#    end
#    def start
#      get_output.print "redfst_reset();\n"
#    end
#    def stop
#      get_output.print "redfst_get_all(energyData);\n"
#    end
#    def compute
#      get_output.print '{
#        VALUE results;
#        VALUE cpu;
#        double pkg, pp0, mem, tot;
#        int i;
#        results = rb_hash_new();
#        tot = 0;
#        for(i=0; i < redfst_ncpus(); ++i){
#          cpu = rb_hash_new();
#          pkg = energyData[3*i + 0];
#          pp0 = energyData[3*i + 1];
#          mem = energyData[3*i + 2];
#          tot += pkg + mem;
#          rb_hash_aset(cpu, rb_str_new2("pkg"), rb_float_new(pkg));
#          rb_hash_aset(cpu, rb_str_new2("pp0"), rb_float_new(pp0));
#          rb_hash_aset(cpu, rb_str_new2("mem"), rb_float_new(mem));
#          rb_hash_aset(cpu, rb_str_new2("total"), rb_float_new(pkg + mem));
#          rb_hash_aset(results, INT2FIX(i), cpu);
#        }
#        rb_hash_aset(results, rb_str_new2("total"), rb_float_new(tot));
#        rb_hash_aset(_boast_stats,ID2SYM(rb_intern("energy")), results);
#      }'
#    end
#  end
#  EnergyProbe = EnergyRedfstProbe
end

