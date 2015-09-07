module BOAST

  module MPPARuntime
    include CRuntime

    alias create_targets_old create_targets

    def target_depends
      return [ module_file_object ]
    end

    def multibinary_path
      return "#{base_path}.mpk"
    end

    def io_bin
      return "#{base_path}.binio"
    end

    def comp_bin
      return "#{base_path}.bincomp"
    end

    def io_object
      return "#{base_path}.#{RbConfig::CONFIG["OBJEXT"]}io"
    end

    def comp_object
      return "#{base_path}.#{RbConfig::CONFIG["OBJEXT"]}comp"
    end

    def io_source
      return "#{base_path}.cio"
    end

    def comp_source
      return "#{base_path}.ccomp"
    end

    def set_io
      set_output(@code_io)
    end

    def set_comp
      @code_comp = StringIO::new unless @code_comp
      set_output(@code_comp)
    end

    attr_accessor :code_comp
    attr_accessor :procedure_comp

    def create_targets( linker, ldshared, ldflags, kernel_files )
      create_targets_old( linker, ldshared, ldflags, kernel_files )
      file multibinary_path => [io_bin, comp_bin] do
        sh "k1-create-multibinary --clusters #{comp_bin} --clusters-names \"comp-part\" --boot #{io_bin} --bootname \"io-part\" -T #{multibinary_path}"
      end
      Rake::Task[multibinary_path].invoke
    end

    def fill_multibinary_header
      fill_library_header
      get_output.puts "#include <mppaipc.h>"
      get_output.puts "#include <mppa/osconfig.h>"
    end

    def copy_array_param_from_host( param )
      get_output.write <<EOF
  mppa_read(_mppa_from_host_size, &_mppa_#{param}_size, sizeof(_mppa_#{param}_size));
  #{param} = malloc(_mppa_#{param}_size);
  mppa_read(_mppa_from_host_var, #{param}, _mppa_#{param}_size);
EOF
    end

    def copy_scalar_param_from_host( param )
      get_ouput.write  <<EOF
  mppa_read(_mppa_from_host_var, &#{param}, sizeof(#{param}));
EOF
    end

    def get_cluster_list_from_host
      get_output.write <<EOF
  mppa_read(_mppa_from_host_size, &_mppa_tmp_size, sizeof(_mppa_tmp_size));
  _clust_list = malloc(_mppa_tmp_size);
  _nb_clust = _mppa_tmp_size / sizeof(uint32_t);
  mppa_read(_mppa_from_host_var, _clust_list, _mppa_tmp_size);
EOF
    end

    def fill_multibinary_source(mode)
      fill_multibinary_header
      code = nil
      if mode == :io then
        code = @code
      else
        code = @code_comp
      end
      code.rewind
      get_output.write code.read
      get_output.puts "int main(int argc, const char* argv[]) {"
      if mode == :io then
        #Parameters declaration
        if @architecture == MPPA then
          @procedure.parameters.each { |param|
            get_output.write "  #{param.type.decl}"
            get_output.write "*" if param.dimension
            get_output.puts " #{param.name};"
            if param.dimension then
              get_output.puts " size_t _mppa_#{param.name}_size;"
            end
          }
        end
          
        #Cluster list declaration
        get_output.write <<EOF
  uint32_t* _clust_list;
  int _nb_clust;
EOF

        #Receiving parameters from Host
        get_output.write <<EOF
  int _mppa_from_host_size, _mppa_from_host_var, _mppa_to_host_size, _mppa_to_host_var, _mppa_tmp_size, _mppa_pid[16], i;
  _mppa_from_host_size = mppa_open("/mppa/buffer/board0#mppa0#pcie0#2/host#2", O_RDONLY);
  _mppa_from_host_var = mppa_open("/mppa/buffer/board0#mppa0#pcie0#3/host#3", O_RDONLY);
EOF
        @procedure.parameters.each { |param|
          if param.dimension then
            copy_array_param_from_host(param)
          else
            copy_scalar_param_from_host(param)
          end
        }

        #Receiving cluster list
        get_cluster_list_from_host

        get_output.write <<EOF
  mppa_close(_mppa_from_host_size);
  mppa_close(_mppa_from_host_var);
EOF
        #Spawning cluster
        get_output.write <<EOF
  for(i=0; i<_nb_clust;i++){
    _mppa_pid[i] = mppa_spawn(_clust_list[i], NULL, "comp-part", NULL, NULL);
  }
EOF
        get_output.write "    #{@procedure.name}("
        @procedure.parameters.each_with_index { |param, i|
        get_output.write ", " unless i == 0
          if !param.dimension then
            if param.direction == :out or param.direction == :inout then
              get_output.write "&"
            end
          end
          get_output.write param.name
        }
        get_output.write ");\n"
      else #Compute code
        source_file.write "    #{@procedure_comp.name}();\n"
      end
        
        
      #Sending results to Host
      if mode == :io then #IO Code
        get_output.write <<EOF
  for(i=0; i< _nb_clust; i++){
    mppa_waitpid(_mppa_pid[i], NULL, 0);
  }
  _mppa_to_host_size = mppa_open("/mppa/buffer/host#4/board0#mppa0#pcie0#4", O_WRONLY);
  _mppa_to_host_var = mppa_open("/mppa/buffer/host#5/board0#mppa0#pcie0#5", O_WRONLY);
EOF
        @procedure.parameters.each { |param| 
          if param.direction == :out or param.direction == :inout then
            if param.dimension then
              source_file.write <<EOF
  _mppa_tmp_size = #{param.dimension.size};
  mppa_write(_mppa_to_host_size, &_mppa_tmp_size, sizeof(_mppa_tmp_size));
  mppa_write(_mppa_to_host_var, #{param.name}, _mppa_tmp_size);
EOF
              else
                source_file.write <<EOF
    mppa_write(_mppa_to_host_var, &#{param.name}, sizeof(#{param.name}));
EOF
              end
            end
          }
          source_file.write <<EOF
    mppa_close(_mppa_to_host_size);
    mppa_close(_mppa_to_host_var);
EOF
        else #Compute code
        end
        source_file.write <<EOF
    mppa_exit(0);
    return 0;
}
EOF
    end

    def create_multibinary_source(mode)
      f = File::open(self.send("#{mode}_source"),"w+")
      previous_lang = get_lang
      previous_output = get_output
      set_output(f)
      set_lang(@lang)

      fill_multibinary_source(mode)

      if debug_source? then
        f.rewind
        puts f.read
      end
      set_output(previous_output)
      set_lang(previous_lang)
      f.close
    end

    def create_multibinary_sources
      create_multibinary_source(:io)
      create_multibinary_source(:comp)
    end

    def create_sources
      create_multibinary_sources
      create_module_file_source
    end

  end

end
