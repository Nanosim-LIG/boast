require 'os'
require 'yaml'

module BOAST

  X86 = 1
  ARM = 2

  native_flags = nil

  if OS.mac? then
    native_flags = `sysctl -n machdep.cpu.features`.split
  else
    native_flags = YAML::load(`cat /proc/cpuinfo`)["flags"].upcase.gsub("_",".").split
  end

  MODELS={ "native" => native_flags }
  MODELS.update(X86architectures)
  INSTRUCTIONS = {}
  INSTRUCTIONS.update(X86CPUID_by_name)

  module Intrinsics
    extend PrivateStateAccessor
    INTRINSICS = Hash::new { |h, k| h[k] = Hash::new { |h2, k2| h2[k2] = {} } }

    def supported(intr_symbol, type, type2=nil)
      instruction = intrinsics(intr_symbol, type, type2)
      return false unless instruction
      INSTRUCTIONS[instruction.to_s].each { |flag|
        return true if MODELS[get_model].include?(flag)
      }
      return false
    end

    module_function :supported

    def intrinsics(intr_symbol, type, type2=nil)
      return INTRINSICS[get_architecture][intr_symbol][get_vector_name(type)][get_vector_name(type2)] if type2
      return INTRINSICS[get_architecture][intr_symbol][get_vector_name(type)]
    end

    module_function :intrinsics

    def get_vector_decl_X86( data_type )
      raise "Unsupported vector size on X86: #{data_type.total_size*8}!" unless [64,128,256].include?( data_type.total_size*8 )
      s = "__m#{data_type.total_size*8}"
      case data_type
      when Int
        raise "Unsupported data size for int vector on X86: #{data_type.size*8}!" unless [1,2,4,8].include?( data_type.size )
        return s+= "#{data_type.total_size*8>64 ? "i" : ""}"
      when Real
        return s if data_type.size == 4
        return s += "d" if data_type.size == 8
        raise "Unsupported data size for real vector on X86: #{data_type.size*8}!"
      else
        raise "Unsupported data type #{data_type} for vector!"
      end
    end

    module_function :get_vector_decl_X86

    def get_vector_decl_ARM( data_type )
      raise "Unsupported vector size on ARM: #{data_type.total_size*8}!" unless [64,128].include?( data_type.total_size*8 )
      case data_type
      when Int
        raise "Unsupported data size for int vector on ARM: #{data_type.size*8}!" unless [1,2,4,8].include?( data_type.size )
        return get_vector_name( data_type ).to_s
      when Real
        raise "Unsupported data size for real vector on ARM: #{data_type.size*8}!" if data_type.size != 4
        return get_vector_name( data_type ).to_s
      else
        raise "Unsupported data type #{data_type} for vector on ARM!"
      end
    end

    module_function :get_vector_decl_ARM

    def get_vector_decl( data_type )
      case get_architecture
      when X86
        get_vector_decl_X86( data_type )
      when ARM
        get_vector_decl_ARM( data_type )
      else
        return get_vector_name( data_type )
      end
    end

    module_function :get_vector_decl

    def get_vector_name( type )
      s = ""
      case type
      when Int
        s += "u" if type.signed?
        s += "int"
      when Real
        s += "float"
      else
        raise "Undefined vector type!"
      end
      s += "#{type.size*8}"
      s += "x#{type.vector_length}_t"
      return s.to_sym
    end

    module_function :get_vector_name

    def vector_type_name( type, size, vector_size, sign = :signed )
      s = ""
      case type
      when :int
        case sign
        when :signed
          s += "int"
        when :unsigned
          s += "uint"
        else
          raise "Invalid sign!"
        end
      when :float
        s += "float"
      else
        raise "Invalid type!"
      end
      s += "#{size}"
      s += "x#{vector_size/size}_t"
      return s.to_sym
    end

    module_function :vector_type_name

    def type_name_ARM( type, size, sign = :signed )
      s = ""
      case type
      when :int
        case sign
        when :signed
          s += "s"
        when :unsigned
          s += "u"
        else
          raise "Invalid sign!"
        end
      when :float
        s += "f"
      else
        raise "Invalid type!"
      end
      s += "#{size}"
      return s
    end

    module_function :type_name_ARM

    def type_name_X86( type, size, vector_size, sign = :signed )
      s = ""
      e = ( vector_size > 64 ? "e" : "" )
      case type
      when :int
        s += "#{e}p"
        case sign
        when :signed
          s += "i"
        when :unsigned
          s += "u"
        else
          raise "Invalid sign!"
        end
        s += "#{size}"
      when :float
        s += "p"
        case size
        when 32
          s += "s"
        when 64
          s += "d"
        else
          raise "Invalid size!"
        end
      else
        raise "Invalid type!"
      end
      return s
    end

    module_function :type_name_X86

    [64, 128, 256].each { |vector_size|
      vs = ( vector_size < 256 ? "" : "#{vector_size}" )
      sizes = [8, 16, 32]
      sizes.push( 64 ) if vector_size > 64
      sizes.each { |size|
        [:signed, :unsigned].each { |sign|
          vtype = vector_type_name( :int, size, vector_size, sign )
          type = type_name_X86( :int, size, vector_size )
          instructions = [[:ADD, "add"], [:SUB, "sub"]]
          instructions.push( [:SET, "setr"], [:SET1, "set1"] )
          instructions.push( [:MUL, "mullo"] ) if vector_size > 64 and size >= 16 and  size <= 32
          instructions.each { |cl, ins|
            INTRINSICS[X86][cl][vtype] = "_mm#{vs}_#{ins}_#{type}".to_sym
          }
          if size == 64 and vector_size < 512 then
            INTRINSICS[X86][:SET1][vtype] = "_mm#{vs}_set1_#{type}x".to_sym
          end
        }
      }
      [8, 16, 32, 64].each { |size|
        [:signed, :unsigned].each { |sign|
          vtype = vector_type_name( :int, size, vector_size, sign )
          [[:LOAD, "loadu"], [:LOADA, "load"],
           [:STORE, "storeu"], [:STOREA, "store"]].each { |cl, ins|
            INTRINSICS[X86][cl][vtype] = "_mm#{vs}_#{ins}_si#{vector_size}".to_sym
          }
        }
      } if vector_size > 64
      sizes = []
      sizes.push( 32, 64 ) if vector_size > 64
      sizes.each { |size|
        [[:ADD, "add"],       [:SUB, "sub"],           [:MUL, "mul"],       [:DIV, "div"],
         [:FMADD, "fmadd"],   [:FMSUB, "fmsub"],       [:FNMADD, "fnmadd"], [:FNMSUB, "fnmsub"],
         [:ADDSUB, "addsub"], [:FMADDSUB, "fmaddsub"], [:FMSUBADD, "fmsubadd"],
         [:LOAD, "loadu"],    [:LOADA, "load"],
         [:STORE, "storeu"],  [:STOREA, "store"],
         [:SET, "set"],       [:SET1, "set1"] ].each { |cl, ins|
          vtype = vector_type_name( :float, size, vector_size)
          type = type_name_X86( :float, size, vector_size )
          INTRINSICS[X86][cl][vtype] = "_mm#{vs}_#{ins}_#{type}".to_sym
        }
      }
    }
    INTRINSICS[X86][:CVT] = Hash::new { |h,k| h[k] = {} }
    [128, 256].each { |bvsize|
      [16, 32, 64].each { |bsize|
        ssize = bsize/2
        while ssize >= 8
          svsize = (bvsize/bsize)*ssize
          [:signed, :unsigned].each { |sign|
            stype = type_name_X86( :int, ssize, 128,  sign )
            btype = type_name_X86( :int, bsize, bvsize, :signed )
            svtype = vector_type_name( :int, ssize, svsize, sign )
            bvtype = vector_type_name( :int, bsize, bvsize, :signed )
            vs = ( bvsize < 256 ? "" : "#{bvsize}" )
            INTRINSICS[X86][:CVT][bvtype][svtype] = "_mm#{vs}_cvt#{stype}_#{btype}".to_sym
          }
          ssize /= 2
        end
      }
    }


    [64, 128].each { |vector_size|
      q = (vector_size == 128 ? "q" : "")
      [8, 16, 32, 64].each { |size|
        [:signed, :unsigned].each { |sign|
          vtype = vector_type_name( :int, size, vector_size, sign )
          type = type_name_ARM( :int, size, sign )
          instructions = [[:ADD, "add"], [:SUB, "sub"]]
          instructions.push( [:MUL, "mul"], [:FMADD, "mla"], [:FNMSUB, "mls"] ) if size < 64
          instructions.push( [:LOAD, "ldl"], [:LOADA, "ldl"] )
          instructions.push( [:STORE, "stl"], [:STOREA, "stl"] )
          instructions.each { |cl, ins|
            INTRINSICS[ARM][cl][vtype] = "v#{ins}#{q}_#{type}".to_sym
          }
          [[:SET1, "dup"]].each { |cl, ins|
            INTRINSICS[ARM][cl][vtype] = "v#{ins}#{q}_n_#{type}".to_sym
          }
          [[:SET_LANE, "set"]].each { |cl, ins|
            INTRINSICS[ARM][cl][vtype] = "v#{ins}#{q}_lane_#{type}".to_sym
          }
        }
      }
      [32].each { |size|
        vtype = vector_type_name( :float, size, vector_size )
        type = type_name_ARM( :float, size )
        [[:ADD, "add"], [:SUB, "sub"], [:MUL, "mul"],
         [:FMADD, "mla"], [:FNMSUB, "mls"],
         [:LOAD, "ldl"], [:LOADA, "ldl"],
         [:STORE, "stl"], [:STOREA, "stl"]].each { |cl, ins|
          INTRINSICS[ARM][cl][vtype] = "v#{ins}#{q}_#{type}".to_sym
        }
        [[:SET1, "dup"]].each { |cl, ins|
          INTRINSICS[ARM][cl][vtype] = "v#{ins}#{q}_n_#{type}".to_sym
        }
      }
    }
    INTRINSICS[ARM][:CVT] = Hash::new { |h,k| h[k] = {} }
    [64, 128].each { |vector_size|
      int_size = 32
      float_size = 32
      q = (vector_size == 128 ? "q" : "")
      [:signed, :unsigned].each { |sign|
        fvtype = vector_type_name( :float, float_size, vector_size )
        ivtype = vector_type_name( :int, int_size, vector_size, sign )
        ftype = type_name_ARM( :float, float_size )
        itype = type_name_ARM( :int, int_size, sign )
        INTRINSICS[ARM][:CVT][fvtype][ivtype] = "vcvt#{q}_#{itype}_#{ftype}".to_sym
        INTRINSICS[ARM][:CVT][ivtype][fvtype] = "vcvt#{q}_#{ftype}_#{itype}".to_sym
      }
    }
    svsize = 64
    bvsize = 128
    [16, 32, 64].each { |bsize|
      ssize = bsize/2
      [:signed, :unsigned].each { |sign|
        stype = type_name_ARM( :int, ssize, sign )
        btype = type_name_ARM( :int, bsize, sign )
        svtype = vector_type_name( :int, ssize, svsize, sign )
        bvtype = vector_type_name( :int, bsize, bvsize, sign )
        INTRINSICS[ARM][:CVT][svtype][bvtype] = "vmovl_#{stype}".to_sym
        INTRINSICS[ARM][:CVT][bvtype][svtype] = "vmovn_#{btype}".to_sym
      }
    }

  end

end
