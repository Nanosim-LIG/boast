module BOAST
  module RedfstProbe
    extend PrivateStateAccessor
    module_function
    def header
      get_output.puts "#include <redfst.h>"
    end
    def decl
      get_output.puts "redfst_dev_t *_boast_energy=0;"
    end
    def configure
      get_output.puts "redfst_init();"
    end
    def start
      get_output.puts "redfst_reset();"
    end
    def stop
      get_output.puts "_boast_energy = redfst_get(_boast_energy);"
    end
    def compute
      get_output.print <<EOF
{
  VALUE results;
  double pkg, pp0, dram;
  char *s;
  int i;
  pkg = pp0 = dram = 0;
  results = rb_hash_new();
  for(i=0; i < _boast_energy->count; ++i){
    rb_hash_aset(results, ID2SYM(rb_intern(_boast_energy->name[i])), rb_float_new(_boast_energy->energy[i]));
    s = _boast_energy->name[i];
    while('.'!=*s++)
      ;
    while('.'!=*s++)
      ;
    if(!strcmp("pkg",s))
      pkg += _boast_energy->energy[i];
    else if(!strcmp("pp0",s))
      pp0 += _boast_energy->energy[i];
    else if(!strcmp("dram",s))
      dram += _boast_energy->energy[i];
  }
  rb_hash_aset(results, ID2SYM(rb_intern("total.pkg" )), rb_float_new(pkg));
  rb_hash_aset(results, ID2SYM(rb_intern("total.pp0" )), rb_float_new(pp0));
  rb_hash_aset(results, ID2SYM(rb_intern("total.dram")), rb_float_new(dram));
  rb_hash_aset(results, ID2SYM(rb_intern("total"     )), rb_float_new(pkg+dram));
  rb_hash_aset(_boast_stats, ID2SYM(rb_intern("energy")), results);
}
EOF
		end
    def is_available
      [] != ENV['LIBRARY_PATH'].split(':').inject([]){|mem, x| []!=mem ? mem : Dir.glob(x+'/libredfst.so')}
    end
  end

  module EmlProbe
    extend PrivateStateAccessor
    module_function
    def header
      get_output.puts "#include <eml.h>"
    end
    def decl
      get_output.puts "emlData_t **_boast_energy=0;";
      get_output.puts "size_t _boast_energy_count=0;";
    end
    def configure
      get_output.puts "emlInit();"
      get_output.puts "emlDeviceGetCount(&_boast_energy_count);"
      get_output.puts "_boast_energy = malloc(_boast_energy_count*sizeof(*_boast_energy));"
    end
    def start
      get_output.puts "emlStart();";
    end
    def stop
      get_output.puts "emlStop(_boast_energy);";
    end
    def compute
      get_output.print <<EOF
{
  VALUE results;
  double consumed;
  results = rb_hash_new();
  emlDataGetConsumed(_boast_energy[0], &consumed);
  rb_hash_aset(results, ID2SYM(rb_intern("total" )), rb_float_new(consumed));
  rb_hash_aset(_boast_stats, ID2SYM(rb_intern("energy")), results);
}
EOF
    end
    def is_available
      [] != ENV['LIBRARY_PATH'].split(':').inject([]){|mem, x| []!=mem ? mem : Dir.glob(x+'/libeml.so')}
    end
  end
  if RedfstProbe.is_available
    EnergyProbe = RedfstProbe
  elsif EmlProbe.is_available
    EnergyProbe = EmlProbe
  else
    EnergyProbe = 'no energy probe available'
  end
end
