module BOAST
  module OpenCLRuntime
    OPENCL_REAL_TYPES = {
      2 => OpenCL::Half1,
      4 => OpenCL::Float1,
      8 => OpenCL::Double1
    }

    OPENCL_INT_TYPES = {
      true => {
        1 => OpenCL::Char1,
        2 => OpenCL::Short1,
        4 => OpenCL::Int1,
        8 => OpenCL::Long1
      },
      false => {
        1 => OpenCL::UChar1,
        2 => OpenCL::UShort1,
        4 => OpenCL::UInt1,
        8 => OpenCL::ULong1
      }
    }
  end
end
