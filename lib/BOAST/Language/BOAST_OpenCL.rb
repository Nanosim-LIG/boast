module BOAST

  module OpenCLHelper

    OCL_CUDA_DIM_ASSOC = { 0 => "x", 1 => "y", 2 => "z" }

    CUDA_THREADIDX = BOAST::CStruct("threadIdx",:type_name => "cuda_trheadIdx", :members => [BOAST::Int("x", :signed => false),BOAST::Int("y", :signed => false),BOAST::Int("z", :signed => false)])
    CUDA_BLOCKIDX = BOAST::CStruct("blockIdx",:type_name => "cuda_blockIdx", :members => [BOAST::Int("x", :signed => false),BOAST::Int("y", :signed => false),BOAST::Int("z", :signed => false)])
    CUDA_BLOCKDIM = BOAST::CStruct("blockDim",:type_name => "cuda_blockDim", :members => [BOAST::Int("x", :signed => false),BOAST::Int("y", :signed => false),BOAST::Int("z", :signed => false)])
    CUDA_GRIDDIM = BOAST::CStruct("gridDim",:type_name => "cuda_gridDim", :members => [BOAST::Int("x", :signed => false),BOAST::Int("y", :signed => false),BOAST::Int("z", :signed => false)])

    OCL_HIP_DIM_ASSOC = { 0 => "x", 1 => "y", 2 => "z" }

    HIP_THREADIDX = BOAST::CStruct("threadIdx",:type_name => "hip_trheadIdx", :members => [BOAST::Int("x", :signed => false),BOAST::Int("y", :signed => false),BOAST::Int("z", :signed => false)])
    HIP_BLOCKIDX = BOAST::CStruct("blockIdx",:type_name => "hip_blockIdx", :members => [BOAST::Int("x", :signed => false),BOAST::Int("y", :signed => false),BOAST::Int("z", :signed => false)])
    HIP_BLOCKDIM = BOAST::CStruct("blockDim",:type_name => "hip_blockDim", :members => [BOAST::Int("x", :signed => false),BOAST::Int("y", :signed => false),BOAST::Int("z", :signed => false)])
    HIP_GRIDDIM = BOAST::CStruct("gridDim",:type_name => "hip_gridDim", :members => [BOAST::Int("x", :signed => false),BOAST::Int("y", :signed => false),BOAST::Int("z", :signed => false)])



    private_constant :OCL_CUDA_DIM_ASSOC, :CUDA_THREADIDX, :CUDA_BLOCKIDX, :CUDA_BLOCKDIM, :CUDA_GRIDDIM
    private_constant :OCL_HIP_DIM_ASSOC, :HIP_THREADIDX, :HIP_BLOCKIDX, :HIP_BLOCKDIM, :HIP_GRIDDIM


    def barrier(*locality)
      if lang == CL then
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
      elsif lang == HIP then
        return FuncCall::new("__syncthreads")
      else
        raise "Unsupported language!"
      end
    end

    def get_work_dim
      if lang == CL then
        return FuncCall::new("get_work_dim", :return => Int("wd", :signed => false))
      else
        raise "Unsupported language!"
      end
    end
  
    def get_global_size(dim)
      case lang
      when CL
        return FuncCall::new("get_global_size", dim, :return => Sizet)
      when CUDA
        d = OCL_CUDA_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "CUDA_GRIDDIM.#{d}*CUDA_BLOCKDIM.#{d}"
      when HIP
        d = OCL_HIP_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "HIP_GRIDDIM.#{d}*HIP_BLOCKDIM.#{d}"
      else
        raise "Unsupported language!"
      end
    end

    def get_global_id(dim)
      case lang
      when CL
        return FuncCall::new("get_global_id",dim, :return => Sizet)
      when CUDA
        d = OCL_CUDA_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "CUDA_THREADIDX.#{d}+CUDA_BLOCKIDX.#{d}*CUDA_BLOCKDIM.#{d}"
      when HIP
        d = OCL_HIP_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "HIP_THREADIDX.#{d}+HIP_BLOCKIDX.#{d}*HIP_BLOCKDIM.#{d}"
      else
        raise "Unsupported language!"
      end
    end

    def get_local_size(dim)
      case lang
      when CL
        return FuncCall::new("get_local_size",dim, :return => Sizet)
      when CUDA
        d = OCL_CUDA_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "CUDA_BLOCKDIM.#{d}"
      when HIP
        d = OCL_HIP_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "HIP_BLOCKDIM.#{d}"
      else
        raise "Unsupported language!"
      end
    end

    def get_local_id(dim)
      case lang
      when CL
        return FuncCall::new("get_local_id",dim, :return => Sizet)
      when CUDA
        d = OCL_CUDA_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "CUDA_THREADIDX.#{d}"
      when HIP
        d = OCL_HIP_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "HIP_THREADIDX.#{d}"
      else
        raise "Unsupported language!"
      end
    end
  
    def get_num_groups(dim)
      case lang
      when CL
        return FuncCall::new("get_num_groups",dim, :return => Sizet)
      when CUDA
        d = OCL_CUDA_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "CUDA_GRIDDIM.#{d}"
      when HIP
        d = OCL_HIP_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "HIP_GRIDDIM.#{d}"
      else
        raise "Unsupported language!"
      end
    end

    def get_group_id(dim)
      case lang
      when CL
        return FuncCall::new("get_group_id",dim, :return => Sizet)
      when CUDA
        d = OCL_CUDA_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "CUDA_BLOCKIDX.#{d}"
      when HIP
        d = OCL_HIP_DIM_ASSOC[dim]
        raise "Unsupported dimension!" unless d
        return eval "HIP_BLOCKIDX.#{d}"
      else
        raise "Unsupported language!"
      end
    end

  end

  extend OpenCLHelper

  EXTENDED.push OpenCLHelper

end
