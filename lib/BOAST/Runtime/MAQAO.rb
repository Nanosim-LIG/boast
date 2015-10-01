module BOAST

  module CompiledRuntime
    def maqao_analysis(options={})
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

      if verbose? then
        puts "#{compiler_options[:MAQAO]} cqa #{f1.path} --fct=#{@procedure.name} #{compiler_options[:MAQAO_FLAGS]}"
      end
      result = `#{compiler_options[:MAQAO]} cqa #{f1.path} --fct=#{@procedure.name} #{compiler_options[:MAQAO_FLAGS]}`
      File::unlink(library_object)
      File::unlink(library_source)
      return result
    end
  end

end
