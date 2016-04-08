module BOAST

  module CompiledRuntime
    def maqao_analysis(options={})
      maqao_models = {
        "core2" => "CORE2_45", 
        "nehalem" => "NEHALEM",
        "sandybridge" => "SANDY_BRIDGE",
        "ivybridge" => "IVY_BRIDGE",
        "haswell" => "HASWELL"
      }
      compiler_options = BOAST::get_compiler_options
      compiler_options.update(options)

      f1 = File::open(library_object,"wb")
      @binary.rewind
      f1.write( @binary.read )
      f1.close

      f2 = File::open(library_source,"wb")
      @source.rewind
      f2.write( @source.read )
      f2.close
      maqao_model = maqao_models[get_model]
      if verbose? then
        puts "#{compiler_options[:MAQAO]} cqa #{maqao_model ? "--uarch=#{maqao_model} " : ""}#{f1.path} --fct=#{@procedure.name} #{compiler_options[:MAQAO_FLAGS]}"
      end
      result = `#{compiler_options[:MAQAO]} cqa #{maqao_model ? "--uarch=#{maqao_model} " : ""}#{f1.path} --fct=#{@procedure.name} #{compiler_options[:MAQAO_FLAGS]}`
      File::unlink(library_object) unless keep_temp
      File::unlink(library_source) unless keep_temp
      return result
    end
  end

end
