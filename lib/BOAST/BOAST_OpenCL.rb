module BOAST
  @@ocl_cuda_dim_assoc = { 0 => "x", 1 => "y", 2 => "z" }

  @@cuda_threadIdx = CStruct("threadIdx",:type_name => "cuda_trheadIdx", :members => [Int("x", :signed => false),Int("y", :signed => false),Int("z", :signed => false)])
  @@cuda_blockIdx = CStruct("blockIdx",:type_name => "cuda_blockIdx", :members => [Int("x", :signed => false),Int("y", :signed => false),Int("z", :signed => false)])
  @@cuda_blockDim = CStruct("blockDim",:type_name => "cuda_blockDim", :members => [Int("x", :signed => false),Int("y", :signed => false),Int("z", :signed => false)])
  @@cuda_gridDim = CStruct("gridDim",:type_name => "cuda_gridDim", :members => [Int("x", :signed => false),Int("y", :signed => false),Int("z", :signed => false)])

  def self.barrier(*locality)
    if @@lang == CL then
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
    elsif @@lang == CUDA then
      return FuncCall::new("__syncthreads")
    else
      raise "Unsupported language!"
    end
  end


  def self.get_work_dim
    if @@lang == CL then
      return FuncCall::new("get_work_dim", :returns => Int("wd", :signed => false))
    else
      raise "Unsupported language!"
    end
  end
  
  def self.get_global_size(dim)
    if @@lang == CL then
      return FuncCall::new("get_global_size", dim, :returns => Sizet)
    elsif @@lang == CUDA then
      d = @@ocl_cuda_dim_assoc[dim]
      raise "Unsupported dimension!" if not d
      return eval "@@cuda_gridDim.#{d}*@@cuda_blockDim.#{d}"
    else
      raise "Unsupported language!"
    end
  end

  def self.get_global_id(dim)
    if @@lang == CL then
      return FuncCall::new("get_global_id",dim, :returns => Sizet)
    elsif @@lang == CUDA then
      d = @@ocl_cuda_dim_assoc[dim]
      raise "Unsupported dimension!" if not d
      return eval "@@cuda_threadIdx.#{d}+@@cuda_blockIdx.#{d}*@@cuda_blockDim.#{d}"
    else
      raise "Unsupported language!"
    end
  end

  def self.get_local_size(dim)
    if @@lang == CL then
      return FuncCall::new("get_local_size",dim, :returns => Sizet)
    elsif @@lang == CUDA then
      d = @@ocl_cuda_dim_assoc[dim]
      raise "Unsupported dimension!" if not d
      return eval "@@cuda_blockDim.#{d}"
    else
      raise "Unsupported language!"
    end
  end

  def self.get_local_id(dim)
    if @@lang == CL then
      return FuncCall::new("get_local_id",dim, :returns => Sizet)
    elsif @@lang == CUDA then
      d = @@ocl_cuda_dim_assoc[dim]
      raise "Unsupported dimension!" if not d
      return eval "@@cuda_threadIdx.#{d}"
    else
      raise "Unsupported language!"
    end
  end
  
  def self.get_num_groups(dim)
    if @@lang == CL then
      return FuncCall::new("get_num_groups",dim, :returns => Sizet)
    elsif @@lang == CUDA then
      d = @@ocl_cuda_dim_assoc[dim]
      raise "Unsupported dimension!" if not d
      return eval "@@cuda_gridDim.#{d}"
    else
      raise "Unsupported language!"
    end
  end

  def self.get_group_id(dim)
    if @@lang == CL then
      return FuncCall::new("get_group_id",dim, :returns => Sizet)
    elsif @@lang == CUDA then
      d = @@ocl_cuda_dim_assoc[dim]
      raise "Unsupported dimension!" if not d
      return eval "@@cuda_blockIdx.#{d}"
    else
      raise "Unsupported language!"
    end
  end
  
end
