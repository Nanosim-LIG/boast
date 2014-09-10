module BOAST

  OCL_CUDA_DIM_ASSOC = { 0 => "x", 1 => "y", 2 => "z" }

  CUDA_THREADIDX = CStruct("threadIdx",:type_name => "cuda_trheadIdx", :members => [Int("x", :signed => false),Int("y", :signed => false),Int("z", :signed => false)])
  CUDA_BLOCKIDX = CStruct("blockIdx",:type_name => "cuda_blockIdx", :members => [Int("x", :signed => false),Int("y", :signed => false),Int("z", :signed => false)])
  CUDA_BLOCKDIM = CStruct("blockDim",:type_name => "cuda_blockDim", :members => [Int("x", :signed => false),Int("y", :signed => false),Int("z", :signed => false)])
  CUDA_GRIDDIM = CStruct("gridDim",:type_name => "cuda_gridDim", :members => [Int("x", :signed => false),Int("y", :signed => false),Int("z", :signed => false)])

  module_function

  def barrier(*locality)
    if lang == CL then
      loc=""
      if locality.include?(:local) and locality.include?(:global) then
        return FuncCall::new("barrier","CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE")
      elsif locality.include?(:local) then
        return FuncCall::new("barrier","CLK_LOCAL_MEM_FENCE")
      elsif locality.include?(:global) then
        return FuncCall::new("barrier","CLK_GLOBAL_MEM_FENCE")
      else
        raise "Unsupported locality"
      end
    elsif lang == CUDA then
      return FuncCall::new("__syncthreads")
    else
      raise "Unsupported language!"
    end
  end


  def get_work_dim
    if lang == CL then
      return FuncCall::new("get_work_dim", :returns => Int("wd", :signed => false))
    else
      raise "Unsupported language!"
    end
  end
  
  def get_global_size(dim)
    if lang == CL then
      return FuncCall::new("get_global_size", dim, :returns => Sizet)
    elsif lang == CUDA then
      d = OCL_CUDA_DIM_ASSOC[dim]
      raise "Unsupported dimension!" if not d
      return eval "CUDA_GRIDDIM.#{d}*CUDA_BLOCKDIM.#{d}"
    else
      raise "Unsupported language!"
    end
  end

  def get_global_id(dim)
    if lang == CL then
      return FuncCall::new("get_global_id",dim, :returns => Sizet)
    elsif lang == CUDA then
      d = OCL_CUDA_DIM_ASSOC[dim]
      raise "Unsupported dimension!" if not d
      return eval "CUDA_THREADIDX.#{d}+CUDA_BLOCKIDX.#{d}*CUDA_BLOCKDIM.#{d}"
    else
      raise "Unsupported language!"
    end
  end

  def get_local_size(dim)
    if lang == CL then
      return FuncCall::new("get_local_size",dim, :returns => Sizet)
    elsif lang == CUDA then
      d = OCL_CUDA_DIM_ASSOC[dim]
      raise "Unsupported dimension!" if not d
      return eval "CUDA_BLOCKDIM.#{d}"
    else
      raise "Unsupported language!"
    end
  end

  def get_local_id(dim)
    if lang == CL then
      return FuncCall::new("get_local_id",dim, :returns => Sizet)
    elsif lang == CUDA then
      d = OCL_CUDA_DIM_ASSOC[dim]
      raise "Unsupported dimension!" if not d
      return eval "CUDA_THREADIDX.#{d}"
    else
      raise "Unsupported language!"
    end
  end
  
  def get_num_groups(dim)
    if lang == CL then
      return FuncCall::new("get_num_groups",dim, :returns => Sizet)
    elsif lang == CUDA then
      d = OCL_CUDA_DIM_ASSOC[dim]
      raise "Unsupported dimension!" if not d
      return eval "CUDA_GRIDDIM.#{d}"
    else
      raise "Unsupported language!"
    end
  end

  def get_group_id(dim)
    if lang == CL then
      return FuncCall::new("get_group_id",dim, :returns => Sizet)
    elsif lang == CUDA then
      d = OCL_CUDA_DIM_ASSOC[dim]
      raise "Unsupported dimension!" if not d
      return eval "CUDA_BLOCKIDX.#{d}"
    else
      raise "Unsupported language!"
    end
  end

end
