module BOAST

  class CKernel

    def load_ref_inputs(path = "", suffix = ".in" )
      return load_ref_files( path, suffix, :in )
    end

    def load_ref_outputs(path = "", suffix = ".out" )
      return load_ref_files( path, suffix, :out )
    end

    def compare_ref(ref_outputs, outputs, epsilon = nil)
      res = {}
      @procedure.parameters.each_with_index { |param, indx|
        if param.direction == :in or param.constant then
          next
        end
        if param.dimension then
          diff = (outputs[indx] - ref_outputs[indx]).abs
          if epsilon then
            diff.each { |elem|
              raise "Error: #{param.name} different from ref by: #{elem}!" if elem > epsilon
            }
          end
          res[param.name] = diff.max
        else
          raise "Error: #{param.name} different from ref: #{outputs[indx]} != #{ref_outputs[indx]} !" if epsilon and (outputs[indx] - ref_outputs[indx]).abs > epsilon
          res[param.name] = (outputs[indx] - ref_outputs[indx]).abs
        end
      }
      return res
    end

    def get_array_type(param)
      if param.type.class == Real then
        case param.type.size
        when 4
          type = NArray::SFLOAT
        when 8
          type = NArray::FLOAT
        else
          STDERR::puts "Unsupported Float size for NArray: #{param.type.size}, defaulting to byte" if debug?
          type = NArray::BYTE
        end
      elsif param.type.class == Int then
        case param.type.size
        when 1
          type = NArray::BYTE
        when 2
          type = NArray::SINT
        when 4
          type = NArray::SINT
        else
          STDERR::puts "Unsupported Int size for NArray: #{param.type.size}, defaulting to byte" if debug?
          type = NArray::BYTE
        end
      else
        STDERR::puts "Unkown array type for NArray: #{param.type}, defaulting to byte" if debug?
        type = NArray::BYTE
      end
      return type
    end

    def get_scalar_type(param)
      if param.type.class == Real then
        case param.type.size
        when 4
          type = "f"
        when 8
          type = "d"
        else
          raise "Unsupported Real scalar size: #{param.type.size}!"
        end
      elsif param.type.class == Int then
        case param.type.size
        when 1
          type = "C"
        when 2
          type = "S"
        when 4
          type = "L"
        when 8
          type = "Q"
        else
          raise "Unsupported Int scalar size: #{param.type.size}!"
        end
        if param.type.signed? then
          type.downcase!
        end
      end
      return type
    end

    def read_param(param, directory, suffix, intent)
      if intent == :out and ( param.direction == :in or param.constant ) then
        return nil
      end
      f = File::new( directory + "/" + param.name+suffix, "rb" )
      if param.dimension then
        type = get_array_type(param)
        if f.size == 0 then
          res = NArray::new(type, 1)
        else
          res = NArray.to_na(f.read, type)
        end
      else
        type = get_scalar_type(param)
        res = f.read.unpack(type).first
      end
      f.close
      return res
    end

    def get_gpu_dim(directory)
      f = File::new( directory + "/problem_size", "r")
      s = f.read
      local_dim, global_dim = s.scan(/<(.*?)>/)
      local_dim  = local_dim.pop.split(",").collect!{ |e| e.to_i }
      global_dim = global_dim.pop.split(",").collect!{ |e| e.to_i }
      (local_dim.length..2).each{ |i| local_dim[i] = 1 }
      (global_dim.length..2).each{ |i| global_dim[i] = 1 }
      if @lang == CL then
        local_dim.each_index { |indx| global_dim[indx] *= local_dim[indx] }
        res = { :global_work_size => global_dim, :local_work_size => local_dim }
      else
        res = { :block_number => global_dim, :block_size => local_dim }
      end
      f.close
      return res
    end

    def load_ref_files(  path = "", suffix = "", intent )
      proc_path = path + "/#{@procedure.name}/"
      res_h = {}
      begin
        dirs = Pathname.new(proc_path).children.select { |c| c.directory? }
      rescue
        return res_h
      end
      dirs.collect! { |d| d.to_s }
      dirs.each { |d|
        res = [] 
        @procedure.parameters.collect { |param|
          res.push read_param(param, d, suffix, intent)
        }
        if @lang == CUDA or @lang == CL then
          res.push get_gpu_dim(d)
        end
        res_h[d] =  res
      }
      return res_h
    end

  end

end
