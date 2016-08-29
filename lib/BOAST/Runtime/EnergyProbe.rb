module BOAST
  module PowercapProbe
    extend PrivateStateAccessor
    module_function
    def header
      get_output.puts "#include <stdio.h>"
      get_output.puts "#include <stdint.h>"
    end
    def decl
      get_output.puts "char **_boast_energy_files = 0;"
      get_output.puts "char **_boast_energy_names = 0;"
      get_output.puts "uint64_t *_boast_energy_0  = 0;"
      get_output.puts "uint64_t *_boast_energy_1  = 0;"
      get_output.puts "int _boast_energy_nsensors = 0;"
    end
    def configure
      get_output.print <<EOF
{
  char buf[128];
  char path[128];
  char *s;
  FILE *f;
  int nproc;
  int i;
  if( _boast_energy_nsensors ){
    free(_boast_energy_files);
    for(i=0; i < _boast_energy_nsensors; ++i)
      free(_boast_energy_names[i]);
    free(_boast_energy_names);
    _boast_energy_nsensors = 0;
  }
  _boast_energy_files = malloc(1);
  _boast_energy_names = malloc(1);

  for(nproc = 0; ; ++nproc){
    sprintf(path,"/sys/devices/virtual/powercap/intel-rapl/intel-rapl:%d",nproc);
    sprintf(buf,"%s/energy_uj",path);
    f = fopen(buf, "rt");
    if(!f)
      break;
    i = 0;
    do{
      fclose(f);
      ++_boast_energy_nsensors;
      _boast_energy_files = realloc(_boast_energy_files, _boast_energy_nsensors * sizeof(*_boast_energy_files));
      _boast_energy_names = realloc(_boast_energy_names, _boast_energy_nsensors * sizeof(*_boast_energy_names));
      _boast_energy_files[_boast_energy_nsensors-1] = malloc(128);
      _boast_energy_names[_boast_energy_nsensors-1] = malloc( 16);
      s = _boast_energy_names[_boast_energy_nsensors-1];
      sprintf(_boast_energy_files[_boast_energy_nsensors-1],buf);
      sprintf(buf,"%s/name",path);
      f = fopen(buf, "r");
      buf[fread(buf, 1, sizeof(buf), f)-1] = 0;
      fclose(f);
      sprintf(s, "%d.%s", nproc, buf);

      sprintf(path,"/sys/devices/virtual/powercap/intel-rapl/intel-rapl:%d/intel-rapl:%d:%d",nproc,nproc,i++);
      sprintf(buf,"%s/energy_uj",path);
      f = fopen(buf, "rt");
    }while(f);
  }
  if( ! _boast_energy_nsensors ){
    free( _boast_energy_files );
    free( _boast_energy_names );
  }else{
    _boast_energy_0 = malloc(_boast_energy_nsensors * sizeof(*_boast_energy_0));
    _boast_energy_1 = malloc(_boast_energy_nsensors * sizeof(*_boast_energy_1));
  }
}
EOF
    end
    def start
      get_output.print <<EOF
{
  char buf[32];
  FILE *f;
  int i;
  for(i = 0; i < _boast_energy_nsensors; ++i){
    f = fopen(_boast_energy_files[i], "r");
    fread(buf, 1, sizeof(buf), f);
    fclose(f);
    _boast_energy_0[i] = atoll(buf);
  }
}
EOF
    end
    def stop
      get_output.print <<EOF
{
  char buf[32];
  FILE *f;
  int i;
  for(i = 0; i < _boast_energy_nsensors; ++i){
    f = fopen(_boast_energy_files[i], "r");
    fread(buf, 1, sizeof(buf), f);
    fclose(f);
    _boast_energy_1[i] = atoll(buf);
  }
}
EOF
    end
    def compute
      get_output.print <<EOF
{
  VALUE results;
  int i;
  results = rb_hash_new();
  for(i=0; i < _boast_energy_nsensors; ++i){
    rb_hash_aset(results, ID2SYM(rb_intern(_boast_energy_names[i])), rb_float_new((_boast_energy_1[i] - _boast_energy_0[i]) * 1e-6));
  }
  rb_hash_aset(_boast_stats, ID2SYM(rb_intern("energy")), results);
}
EOF
    end
    def is_available
      [] != Dir.glob( '/sys/class/powercap/intel-rapl:0:0' )
    end
  end

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
      path = '/lib:/usr/lib'
      path = ENV['LIBRARY_PATH'] if ENV.has_key? 'LIBRARY_PATH'
      [] != path.split(':').inject([]){|mem, x| []!=mem ? mem : Dir.glob(x+'/libredfst.so')}
    end
    def get_options
      return :LDFLAGS => '-redfst'
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
      path = '/lib:/usr/lib'
      path = ENV['LIBRARY_PATH'] if ENV.has_key? 'LIBRARY_PATH'
      [] != path.split(':').inject([]){|mem, x| []!=mem ? mem : Dir.glob(x+'/libeml.so')}
    end
    def get_options
      return :LDFLAGS => '-leml'
    end
  end
  if PowercapProbe.is_available
    EnergyProbe = PowercapProbe
  elsif RedfstProbe.is_available
    EnergyProbe = RedfstProbe
  elsif EmlProbe.is_available
    EnergyProbe = EmlProbe
  else
    EnergyProbe = nil
  end
end
