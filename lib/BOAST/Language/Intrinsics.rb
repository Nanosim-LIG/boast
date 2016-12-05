require 'os'
require 'yaml'
require 'rgl/adjacency'
require 'rgl/dijkstra'

module BOAST

  native_flags = []

  if OS.mac? then
    native_flags = `sysctl -n machdep.cpu.features`.split
  else
    yaml_cpuinfo = YAML::load(`cat /proc/cpuinfo`)
    cpuinfo_flags = yaml_cpuinfo["flags"]
    cpuinfo_flags = yaml_cpuinfo["Features"] unless cpuinfo_flags
    if cpuinfo_flags then
      native_flags = cpuinfo_flags.upcase.gsub("_",".").split
    else
      warn "Unable to determine architecture flags for native!"
    end
  end

  MODELS = { "native" => native_flags }
  MODELS.update(X86architectures)
  INSTRUCTIONS = {}
  INSTRUCTIONS.update(X86CPUID_by_name)

  class IntrinsicsError < Error
  end

  # @private
  class InternalIntrinsicsError < Error
  end

  # @private
  module Intrinsics
    extend PrivateStateAccessor
    INTRINSICS = Hash::new { |h, k| h[k] = Hash::new { |h2, k2| h2[k2] = {} } }
    CONVERSIONS = Hash::new { |h, k| h[k] = Hash::new { |h2, k2| h2[k2] = {} } }

    def check_coverage
      ins = []
      INTRINSICS[X86].each { |i,v|
        if i == :CVT then
          v.each { |type1, h|
            h.each { |type2, instr|
              ins.push instr.to_s
            }
          }
        else
          v.each { |type, instr|
            ins.push instr.to_s
          }
        end
      }
      return ins - INSTRUCTIONS.keys
    end

    module_function :check_coverage

    def intrinsics_by_vector_name(intr_symbol, type, type2=nil)
      if type2 then
        instruction = INTRINSICS[get_architecture][intr_symbol][type][type2]
      else
        instruction = INTRINSICS[get_architecture][intr_symbol][type]
      end
      raise IntrinsicsError, "Unsupported operation #{intr_symbol} for #{type}#{type2 ? " and #{type2}" : ""} on #{get_architecture_name}!" unless instruction
      return instruction if get_architecture == ARM
      supported = false
      INSTRUCTIONS[instruction.to_s].each { |cpuid|
        if cpuid.kind_of?( Array ) then
          supported = true if (cpuid - MODELS[get_model.to_s]).empty?
        else
          supported = true if MODELS[get_model.to_s].include?( cpuid )
        end
      }
#      supported = (INSTRUCTIONS[instruction.to_s] & MODELS[get_model.to_s]).size > 0
      if not supported then
        required = ""
        INSTRUCTIONS[instruction.to_s].each { |cpuid|
          required += " or " if required != ""
          if cpuid.kind_of?( Array ) then
            required += "( #{cpuid.join(" and ")} )"
          else
            required += "#{cpuid}"
          end
        }
        raise IntrinsicsError, "Unsupported operation #{intr_symbol} for #{type}#{type2 ? " and #{type2}" : ""} on #{get_model}! (requires #{required})"
      end
      return instruction
    end

    module_function :intrinsics_by_vector_name

    def intrinsics(intr_symbol, type, type2=nil)
      return intrinsics_by_vector_name(intr_symbol, get_vector_name(type), type2 ? get_vector_name(type2) : nil)
    end

    module_function :intrinsics

    def get_conversion_path(type_dest, type_orig)
      conversion_path = CONVERSIONS[get_architecture][get_vector_name(type_dest)][get_vector_name(type_orig)]
      raise IntrinsicsError, "Unavailable conversion from #{get_vector_name(type_orig)} to #{get_vector_name(type_dest)} on #{get_architecture_name}#{get_architecture==X86 ? "(#{get_model})" : "" }!" unless conversion_path
      return conversion_path
    end

    module_function :get_conversion_path

    def get_vector_decl_X86( data_type )
      raise IntrinsicsError, "Unsupported vector size on X86: #{data_type.total_size*8}!" unless [64,128,256,512].include?( data_type.total_size*8 )
      s = "__m#{data_type.total_size*8}"
      case data_type
      when Int
        raise IntrinsicsError, "Unsupported data size for int vector on X86: #{data_type.size*8}!" unless [1,2,4,8].include?( data_type.size )
        return s+= "#{data_type.total_size*8>64 ? "i" : ""}"
      when Real
        return s if data_type.size == 4
        return s += "d" if data_type.size == 8
        raise IntrinsicsError, "Unsupported data size for real vector on X86: #{data_type.size*8}!"
      else
        raise IntrinsicsError, "Unsupported data type #{data_type} for vector on X86!"
      end
    end

    module_function :get_vector_decl_X86

    def get_vector_decl_ARM( data_type )
      raise IntrinsicsError, "Unsupported vector size on ARM: #{data_type.total_size*8}!" unless [64,128].include?( data_type.total_size*8 )
      case data_type
      when Int
        raise IntrinsicsError, "Unsupported data size for int vector on ARM: #{data_type.size*8}!" unless [1,2,4,8].include?( data_type.size )
        return get_vector_name( data_type ).to_s
      when Real
        raise IntrinsicsError, "Unsupported data size for real vector on ARM: #{data_type.size*8}!" unless [4,8].include?( data_type.size )
        return get_vector_name( data_type ).to_s
      else
        raise IntrinsicsError, "Unsupported data type #{data_type} for vector on ARM!"
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
        s += "u" unless type.signed?
        s += "int"
      when Real
        s += "float"
      else
        raise InternalIntrinsicsError, "Undefined vector type!"
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
          raise InternalIntrinsicsError, "Invalid sign!"
        end
      when :float
        s += "float"
      else
        raise InternalIntrinsicsError, "Invalid type!"
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
          raise InternalIntrinsicsError, "Invalid sign!"
        end
      when :float
        s += "f"
      else
        raise InternalIntrinsicsError, "Invalid type!"
      end
      s += "#{size}"
      return s
    end

    module_function :type_name_ARM

    def type_name_X86( type, size, vector_size, sign = :signed )
      s = ""
      case type
      when :int
        e = ( vector_size > 64 ? "e" : "" )
        s += "#{e}p"
        case sign
        when :signed
          s += "i"
        when :unsigned
          s += "u"
        else
          raise InternalIntrinsicsError, "Invalid sign!"
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
          raise InternalIntrinsicsError, "Invalid size!"
        end
      else
        raise InternalIntrinsicsError, "Invalid type!"
      end
      return s
    end

    module_function :type_name_X86

    [64, 128, 256, 512].each { |vector_size|
      vs = ( vector_size < 256 ? "" : "#{vector_size}" )
      sizes = [8, 16, 32]
      sizes.push( 64 ) if vector_size > 64
      sizes.each { |size|
        [:signed, :unsigned].each { |sign|
          vtype = vector_type_name( :int, size, vector_size, sign )
          type = type_name_X86( :int, size, vector_size )
          instructions = [[:ADD, "add"], [:SUB, "sub"]]
          instructions.push( [:SET, "setr"] ) unless size < 32 and vector_size == 512
          instructions.push( [:SET1, "set1"] )
          instructions.push( [:MUL, "mullo"] ) if vector_size > 64 and size >= 16
          instructions.push( [:MASKLOAD,    "maskload"],    [:MASKSTORE, "maskstore"] ) if vector_size <= 256 and vector_size >= 128 and size >= 32
          instructions.push( [:MASK_LOAD,   "mask_loadu"],  [:MASK_STORE,  "mask_storeu"],
                             [:MASK_LOADA,  "mask_load"],   [:MASK_STOREA, "mask_store"],
                             [:MASKZ_LOAD,  "maskz_loadu"],
                             [:MASKZ_LOADA, "maskz_load"], ) if vector_size >= 128 and size >= 32
          instructions.each { |cl, ins|
            INTRINSICS[X86][cl][vtype] = "_mm#{vs}_#{ins}_#{type}".to_sym
          }
          if size == 64 and vector_size == 256 then
            INTRINSICS[X86][:SET1][vtype] = "_mm#{vs}_set1_#{type}x".to_sym
            INTRINSICS[X86][:SET][vtype] = "_mm#{vs}_setr_#{type}x".to_sym
          end
        }
      }
      [8, 16, 32, 64].each { |size|
        [:signed, :unsigned].each { |sign|
          vtype = vector_type_name( :int, size, vector_size, sign )
          instructions = [[:LOAD, "loadu"],   [:LOADA, "load"],
                          [:STORE, "storeu"], [:STOREA, "store"],
                          [:SETZERO, "setzero"] ]
          instructions.each { |cl, ins|
            INTRINSICS[X86][cl][vtype] = "_mm#{vs}_#{ins}_si#{vector_size}".to_sym
          }
        }
      } if vector_size > 64
      sizes = []
      sizes.push( 32, 64 ) if vector_size > 64
      sizes.each { |size|
        instructions = [[:ADD, "add"],           [:SUB, "sub"], [:MUL, "mul"], [:DIV, "div"], [:POW, "pow"],
                        [:FMADD, "fmadd"],       [:FMSUB, "fmsub"],
                        [:FNMADD, "fnmadd"],     [:FNMSUB, "fnmsub"],
                        [:FMADDSUB, "fmaddsub"], [:FMSUBADD, "fmsubadd"],
                        [:LOAD, "loadu"],        [:LOADA, "load"],
                        [:STORE, "storeu"],  [:STOREA, "store"],
                        [:SET, "setr"],      [:SET1, "set1"], [:SETZERO, "setzero"],
                        [:MASK_LOAD,   "mask_loadu"],  [:MASK_STORE,  "mask_storeu"],
                        [:MASK_LOADA,  "mask_load"],   [:MASK_STOREA, "mask_store"],
                        [:MASKZ_LOAD,  "maskz_loadu"],
                        [:MASKZ_LOADA, "maskz_load"],
                        [:MAX, "max"], [:MIN, "min"],
                        [:SQRT, "sqrt"], [:EXP, "exp"], [:LOG, "log"], [:LOG10, "log10"],
                         [:SIN,   "sin"],   [:COS,   "cos"],   [:TAN,   "tan"],
                         [:SINH,  "sinh"],  [:COSH,  "cosh"],  [:TANH,  "tanh"],
                        [:ASIN,  "asin"],  [:ACOS,  "acos"],  [:ATAN,  "atan"],
                        [:ASINH, "asinh"], [:ACOSH, "acosh"], [:ATANH, "atanh"]]
        instructions.push( [:MASKLOAD, "maskload"], [:MASKSTORE, "maskstore"] ) if vector_size < 512
        instructions.push( [:ADDSUB, "addsub"] ) if vector_size < 512
        instructions.each { |cl, ins|
          vtype = vector_type_name( :float, size, vector_size)
          type = type_name_X86( :float, size, vector_size )
          INTRINSICS[X86][cl][vtype] = "_mm#{vs}_#{ins}_#{type}".to_sym
        }
      }
    }
    INTRINSICS[X86][:CVT] = Hash::new { |h,k| h[k] = {} }
    [128, 256, 512].each { |bvsize|
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
      [64].each { |bsize|
        ssize = bsize/2
        svsize = (bvsize/bsize)*ssize
        stype = type_name_X86( :float, ssize, 128 )
        btype = type_name_X86( :float, bsize, bvsize )
        svtype = vector_type_name( :float, ssize, svsize )
        bvtype = vector_type_name( :float, bsize, bvsize )
        vs = ( bvsize < 256 ? "" : "#{bvsize}" )
        INTRINSICS[X86][:CVT][bvtype][svtype] = "_mm#{vs}_cvt#{stype}_#{btype}".to_sym
        INTRINSICS[X86][:CVT][svtype][bvtype] = "_mm#{vs}_cvt#{btype}_#{stype}".to_sym
      }
      [64].each { |fsize|
        [32].each { |isize|
          ftype = type_name_X86( :float, fsize, bvsize )
          itype = type_name_X86( :int, isize, isize*(bvsize/fsize), :signed )
          fvtype = vector_type_name( :float, fsize, bvsize )
          ivtype = vector_type_name( :int, isize, isize*(bvsize/fsize), :signed )
          vs = ( bvsize < 256 ? "" : "#{bvsize}" )
          INTRINSICS[X86][:CVT][fvtype][ivtype] = "_mm#{vs}_cvt#{itype}_#{ftype}".to_sym
          INTRINSICS[X86][:CVT][ivtype][fvtype] = "_mm#{vs}_cvt#{ftype}_#{itype}".to_sym
        }
      }
      [64,32].each { |bsize|
        ftype = type_name_X86( :float, bsize, bvsize )
        itype = type_name_X86( :int,   bsize, bvsize, :signed )
        fvtype = vector_type_name( :float, bsize, bvsize )
        ivtype = vector_type_name( :int,   bsize, bvsize, :signed )
        vs = ( bvsize < 256 ? "" : "#{bvsize}" )
        INTRINSICS[X86][:CVT][fvtype][ivtype] = "_mm#{vs}_cvt#{itype}_#{ftype}".to_sym
        INTRINSICS[X86][:CVT][ivtype][fvtype] = "_mm#{vs}_cvt#{ftype}_#{itype}".to_sym
      }
    }


    [64, 128].each { |vector_size|
      q = (vector_size == 128 ? "q" : "")
      [8, 16, 32, 64].each { |size|
        [:signed, :unsigned].each { |sign|
          vtype = vector_type_name( :int, size, vector_size, sign )
          type = type_name_ARM( :int, size, sign )
          instructions = [[:ADD, "add"], [:SUB, "sub"]]
          instructions.push( [:MUL, "mul"], [:FMADD, "mla"], [:FNMADD, "mls"] ) if size < 64
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
      [32, 64].each { |size|
        vtype = vector_type_name( :float, size, vector_size )
        type = type_name_ARM( :float, size )
        [[:ADD, "add"], [:SUB, "sub"], [:MUL, "mul"],
         [:FMADD, "mla"], [:FNMADD, "mls"],
         [:LOAD, "ldl"], [:LOADA, "ldl"],
         [:STORE, "stl"], [:STOREA, "stl"]].each { |cl, ins|
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
    INTRINSICS[ARM][:CVT] = Hash::new { |h,k| h[k] = {} }
    [64, 128].each { |vector_size|
      [[32, 32],[64, 64]].each { |int_size, float_size|
        q = (vector_size == 128 ? "q" : "")
        [:signed, :unsigned].each { |sign|
          fvtype = vector_type_name( :float, float_size, vector_size )
          ivtype = vector_type_name( :int, int_size, vector_size, sign )
          ftype = type_name_ARM( :float, float_size )
          itype = type_name_ARM( :int, int_size, sign )
          INTRINSICS[ARM][:CVT][ivtype][fvtype] = "vcvt#{q}_#{itype}_#{ftype}".to_sym
          INTRINSICS[ARM][:CVT][fvtype][ivtype] = "vcvt#{q}_#{ftype}_#{itype}".to_sym
        }
      }
    }
    sfvtype = vector_type_name( :float, 32, 64 )
    sdvtype = vector_type_name( :float, 64, 128 )
    sftype = type_name_ARM( :float, 32 )
    sdtype = type_name_ARM( :float, 64 )
    INTRINSICS[ARM][:CVT][sfvtype][sdvtype] = "vcvt_#{sftype}_#{sdtype}".to_sym
    INTRINSICS[ARM][:CVT][sdvtype][sfvtype] = "vcvt_#{sdtype}_#{sftype}".to_sym
    svsize = 64
    bvsize = 128
    [16, 32, 64].each { |bsize|
      ssize = bsize/2
      [:signed, :unsigned].each { |sign|
        stype = type_name_ARM( :int, ssize, sign )
        btype = type_name_ARM( :int, bsize, sign )
        svtype = vector_type_name( :int, ssize, svsize, sign )
        bvtype = vector_type_name( :int, bsize, bvsize, sign )
        INTRINSICS[ARM][:CVT][bvtype][svtype] = "vmovl_#{stype}".to_sym
        INTRINSICS[ARM][:CVT][svtype][bvtype] = "vmovn_#{btype}".to_sym
      }
    }

    def generate_conversions
      [X86, ARM].each { |arch|
        cvt_dgraph = RGL::DirectedAdjacencyGraph::new
        INTRINSICS[arch][:CVT].each { |dest, origs|
          origs.each { |orig, intrinsic|
            supported = true
            if arch == X86 then
              supported = false
              if MODELS[get_model.to_s] then
                INSTRUCTIONS[intrinsic.to_s].each { |cpuid|
                  if cpuid.kind_of?( Array ) then
                    supported = true if (cpuid - MODELS[get_model.to_s]).empty?
                  else
                    supported = true if MODELS[get_model.to_s].include?( cpuid )
                  end
                }
              end
            end
            cvt_dgraph.add_edge(orig, dest) if supported
          }
        }
        cvt_dgraph.vertices.each { |source|
          hash = {}
          cvt_dgraph.edges.each { |e| hash[e.to_a] = 1 }
          paths = cvt_dgraph.dijkstra_shortest_paths( hash, source )
          paths.each { |dest, path|
            CONVERSIONS[arch][dest][source] = path if path
          }
        }
        types = []
        INTRINSICS[arch].each { |intrinsic, instructions|
          types += instructions.keys
        }
        types.uniq
        types.each { |type|
          CONVERSIONS[arch][type][type] = [type]
        }
      }
    end
    module_function :generate_conversions

    generate_conversions

  end

end
