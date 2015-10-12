module BOAST

  NATIVE = 0

  INTRINSICS = { :X86 => Hash::new { |h, k| h[k] = {} }, :ARM => Hash::new { |h, k| h[k] = {} } }


  def self.vector_type_name( type, size, vector_size, sign = :signed )
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
    s += "x#{vector_size/size}"
    return s.to_sym
  end

  def self.type_name_ARM( type, size, sign = :signed )
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

  def self.type_name_X86( type, size, vector_size, sign = :signed )
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

  [64, 128, 256].each { |vector_size|
    vs = ( vector_size < 256 ? "" : "#{vector_size}" )
    sizes = [8, 16, 32]
    sizes.push( 64 ) if vector_size > 64
    sizes.each { |size|
      [:signed, :unsigned].each { |sign|
        vtype = vector_type_name( :int, size, vector_size, sign )
        type = type_name_X86( :int, size, vector_size )
        instructions = [[:ADD, "add"], [:SUB, "sub"]]
        instructions.push( [:SET, "set"], [:SET1, "set1"] )
        instructions.push( [:MULLO, "mullo"] ) if vector_size > 64 and size >= 16 and  size <= 32
        instructions.each { |cl, ins|
          INTRINSICS[:X86][cl][vtype] = "_mm#{vs}_#{ins}_#{type}".to_sym
        }
      }
    }
    [8, 16, 32, 64].each { |size|
      [:signed, :unsigned].each { |sign|
        vtype = vector_type_name( :int, size, vector_size, sign )
        [[:LOAD, "loadu"], [:LOADA, "load"]].each { |cl, ins|
          INTRINSICS[:X86][cl][vtype] = "_mm#{vs}_#{ins}_si#{vector_size}".to_sym
        }
      }
    } if vector_size > 64
    sizes = []
    sizes.push( 32, 64 ) if vector_size > 64
    sizes.each { |size|
      [[:ADD, "add"], [:SUB, "sub"], [:MUL, "mul"], [:DIV, "div"],
       [:FMADD, "fmadd"], [:FMSUB, "fmsub"], [:FNMADD, "fnmadd"], [:FNMSUB, "fnmsub"],
       [:ADDSUB, "addsub"], [:FMADDSUB, "fmaddsub"], [:FMSUBADD, "fmsubadd"],
       [:LOAD, "loadu"], [:LOADA, "load"], [:SET, "set"], [:SET1, "set1"] ].each { |cl, ins|
        vtype = vector_type_name( :float, size, vector_size)
        type = type_name_X86( :float, size, vector_size )
        INTRINSICS[:X86][cl][vtype] = "_mm#{vs}_#{ins}_#{type}".to_sym
      }
    }
  }
  INTRINSICS[:X86][:CVT] = Hash::new { |h,k| h[k] = {} }
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
          INTRINSICS[:X86][:CVT][bvtype][svtype] = "_mm#{vs}_cvt#{stype}_#{btype}".to_sym
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
        instructions.push( [:MULLO, "mul"], [:FMADD, "mla"], [:FNMSUB, "mls"] ) if size < 64
        instructions.push( [:LOAD, "ldl"], [:LOADA, "ldl"] )
        instructions.each { |cl, ins|
          INTRINSICS[:ARM][cl][vtype] = "v#{ins}#{q}_#{type}".to_sym
        }
        [:SET1, "dup"].each { |cl, ins|
          INTRINSICS[:ARM][cl][vtype] = "v#{ins}#{q}_n_#{type}".to_sym
        }
      }
    }
    [32].each { |size|
      vtype = vector_type_name( :float, size, vector_size )
      type = type_name_ARM( :float, size )
      [[:ADD, "add"], [:SUB, "sub"], [:MUL, "mul"],
       [:FMADD, "mla"], [:FNMSUB, "mls"],
       [:LOAD, "ldl"], [:LOADA, "ldl"]].each { |cl, ins|
        INTRINSICS[:ARM][cl][vtype] = "v#{ins}#{q}_#{type}".to_sym
      }
      [:SET1, "dup"].each { |cl, ins|
        INTRINSICS[:ARM][cl][vtype] = "v#{ins}#{q}_n_#{type}".to_sym
      }
    }
  }
  INTRINSICS[:ARM][:CVT] = Hash::new { |h,k| h[k] = {} }
  [64, 128].each { |vector_size|
    int_size = 32
    float_size = 32
    q = (vector_size == 128 ? "q" : "")
    [:signed, :unsigned].each { |sign|
      fvtype = vector_type_name( :float, float_size, vector_size )
      ivtype = vector_type_name( :int, int_size, vector_size, sign )
      ftype = type_name_ARM( :float, float_size )
      itype = type_name_ARM( :int, int_size, sign )
      INTRINSICS[:ARM][:CVT][fvtype][ivtype] = "vcvt#{q}_#{itype}_#{ftype}".to_sym
      INTRINSICS[:ARM][:CVT][ivtype][fvtype] = "vcvt#{q}_#{ftype}_#{itype}".to_sym
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
      INTRINSICS[:ARM][:CVT][svtype][bvtype] = "vmovl_#{stype}".to_sym
      INTRINSICS[:ARM][:CVT][bvtype][svtype] = "vmovn_#{btype}".to_sym
    }
  }
end
