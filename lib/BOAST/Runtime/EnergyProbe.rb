module BOAST
  module PowercapProbe
    extend PrivateStateAccessor
    module_function
    def header
      get_output.puts "#include <stdio.h>"
      get_output.puts "#include <stdint.h>"
    end
    def preamble
      get_output.print <<EOF
struct _boast_powercap_param_struct {
  char **files;
  char **names;
  uint64_t *energy_0;
  uint64_t *energy_1;
  int nsensors;
};

static int _boast_powercap_init(uint64_t **energy_0, uint64_t **energy_1, char ***files, char ***names);
static int _boast_powercap_init(uint64_t **energy_0, uint64_t **energy_1, char ***files, char ***names) {
  int nsensors = 0;
  char buf[128];
  char path[128];
  char *s;
  FILE *f;
  int nproc;
  int i;

  *files = malloc(1);
  *names = malloc(1);
  for(nproc = 0; ; ++nproc){
    sprintf(path,"/sys/devices/virtual/powercap/intel-rapl/intel-rapl:%d",nproc);
    sprintf(buf,"%s/energy_uj",path);
    f = fopen(buf, "rt");
    if(!f)
      break;
    i = 0;
    do {
      int fread_ret;
      fclose(f);
      ++nsensors;
      *files = realloc(*files, nsensors * sizeof(**files));
      *names = realloc(*names, nsensors * sizeof(**names));
      (*files)[nsensors-1] = malloc(128);
      (*names)[nsensors-1] = malloc( 16);
      s = (*names)[nsensors-1];
      sprintf((*files)[nsensors-1],"%s",buf);
      sprintf(buf, "%s/name", path);
      f = fopen(buf, "r");
      fread_ret = fread(buf, 1, sizeof(buf), f);
      fclose(f);
      if(fread_ret == 0)
        rb_raise(rb_eArgError, "Energy probe read error!");
      /* last character read is a line break */
      buf[fread_ret-1] = 0;
      sprintf(s, "%d.%s", nproc, buf);

      sprintf(path,"/sys/devices/virtual/powercap/intel-rapl/intel-rapl:%d/intel-rapl:%d:%d",nproc,nproc,i++);
      sprintf(buf,"%s/energy_uj",path);
      f = fopen(buf, "rt");
    } while(f);
  }
  if( ! nsensors ) {
    free( *files );
    free( *names );
  } else {
    *energy_0 = malloc(nsensors * sizeof(**energy_0));
    *energy_1 = malloc(nsensors * sizeof(**energy_1));
  }
  return nsensors;
}

static void _boast_powercap_read(int nsensors, char **files, uint64_t *energy);
static void _boast_powercap_read(int nsensors, char **files, uint64_t *energy) {
  char buf[32];
  FILE *f;
  int i;
  for(i = 0; i < nsensors; ++i){
    int fread_ret;
    f = fopen(files[i], "r");
    fread_ret = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    if(fread_ret == 0) {
      rb_raise(rb_eArgError, "Energy probe read error!");
    }
    energy[i] = atoll(buf);
  }
}

static void _boast_powercap_store_and_clean( int nsensors, uint64_t *energy_0, uint64_t *energy_1, char **files, char **names, VALUE _boast_stats);
static void _boast_powercap_store_and_clean( int nsensors, uint64_t *energy_0, uint64_t *energy_1, char **files, char **names, VALUE _boast_stats) {
  VALUE results;
  int i;
  if( nsensors ) {
    results = rb_hash_new();
    for(i=0; i < nsensors; ++i){
      rb_hash_aset(results, ID2SYM(rb_intern(names[i])), rb_float_new((energy_1[i] - energy_0[i]) * 1e-6));
    }
    rb_hash_aset(_boast_stats, ID2SYM(rb_intern("energy")), results);

    free(energy_0);
    free(energy_1);
    for(i=0; i < nsensors; ++i){
      free(files[i]);
      free(names[i]);
    }
    free(files);
    free(names);
  }
}

EOF
    end
    def decl
      get_output.puts "  struct _boast_powercap_param_struct _boast_powercap_params = {0,0,0,0,0};"
    end
    def configure
      get_output.print <<EOF
  _boast_powercap_params.nsensors = _boast_powercap_init(&_boast_powercap_params.energy_0, &_boast_powercap_params.energy_1, &_boast_powercap_params.files, &_boast_powercap_params.names);
EOF
    end
    def start
      get_output.print <<EOF
  _boast_powercap_read(_boast_powercap_params.nsensors, _boast_powercap_params.files, _boast_powercap_params.energy_0);
EOF
    end
    def stop
      get_output.print <<EOF
  _boast_powercap_read(_boast_powercap_params.nsensors, _boast_powercap_params.files, _boast_powercap_params.energy_1);
EOF
    end
    def compute
    end
    def store
      get_output.print <<EOF
  _boast_powercap_store_and_clean( _boast_powercap_params.nsensors, _boast_powercap_params.energy_0, _boast_powercap_params.energy_1, _boast_powercap_params.files, _boast_powercap_params.names, _boast_stats);
EOF
    end
    def is_available?
      [] != Dir.glob( '/sys/class/powercap/intel-rapl:0:0' )
    end
  end

  module RedfstProbe
    extend PrivateStateAccessor
    module_function
    def header
      get_output.puts "#include <redfst.h>"
    end
    def preamble
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
    end
    def store
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
    def is_available?
      return false if OS.mac?
      path = []
      if ENV['LIBRARY_PATH'] then
        path += ENV['LIBRARY_PATH'].split(':').inject([]){|mem, x| []!=mem ? mem : Dir.glob(x+'/libredfst.so')}
      end
      begin
        path += `ldconfig -p`.gsub("\t","").split("\n").find_all { |e| e.match(/libredfst\.so/) }.collect { |e| e.split(" => ")[1] } if path == []
      rescue
        path += `/sbin/ldconfig -p`.gsub("\t","").split("\n").find_all { |e| e.match(/libredfst\.so/) }.collect { |e| e.split(" => ")[1] } if path == []
      end
      return path != []
    end
  end

  module EmlProbe
    extend PrivateStateAccessor
    module_function
    def header
      get_output.puts "#include <eml.h>"
    end
    def preamble
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
    end
    def store
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
    def is_available?
      return false if OS.mac?
      path = []
      if ENV['LIBRARY_PATH'] then
        path += ENV['LIBRARY_PATH'].split(':').inject([]){|mem, x| []!=mem ? mem : Dir.glob(x+'/libeml.so')}
      end
      begin
        path += `ldconfig -p`.gsub("\t","").split("\n").find_all { |e| e.match(/libeml\.so/) }.collect { |e| e.split(" => ")[1] } if path == []
      rescue
        path += `/sbin/ldconfig -p`.gsub("\t","").split("\n").find_all { |e| e.match(/libeml\.so/) }.collect { |e| e.split(" => ")[1] } if path == []
      end
      return path != []
    end
  end
  if PowercapProbe.is_available?
    EnergyProbe = PowercapProbe
  elsif RedfstProbe.is_available?
    EnergyProbe = RedfstProbe
  elsif EmlProbe.is_available?
    EnergyProbe = EmlProbe
  else
    EnergyProbe = nil
  end
end
