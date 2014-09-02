module BOAST
  class Procedure
    include BOAST::Inspectable
    def self.parens(*args,&block)
      return self::new(*args,&block)
    end

    attr_reader :name
    attr_reader :parameters
    attr_reader :constants
    attr_reader :properties
    attr_reader :headers
    def initialize(name, parameters=[], constants=[], properties={}, &block)
      @name = name
      @parameters = parameters
      @constants = constants
      @block = block
      @properties = properties
      @headers = properties[:headers]
      @headers = [] if not @headers
    end

    def header(lang=C,final=true)
      s = ""
      headers.each { |h|
        s += "#include <#{h}>\n"
      }
      if BOAST::get_lang == CL then
        s += "__kernel "
        wgs = @properties[:reqd_work_group_size]
        if wgs then
          s += "__attribute__((reqd_work_group_size(#{wgs[0]},#{wgs[1]},#{wgs[2]}))) "
        end
      end
      trailer = ""
      trailer += "_" if lang == FORTRAN
      trailer += "_wrapper" if lang == CUDA
      if @properties[:return] then
        s += "#{@properties[:return].type.decl} "
      elsif lang == CUDA
        s += "unsigned long long int "
      else
        s += "void "
      end
      s += "#{@name}#{trailer}("
      if parameters.first then
        s += parameters.first.header(lang,false)
        parameters[1..-1].each { |p|
          s += ", "
          s += p.header(lang,false)
        }
      end
      if lang == CUDA then
        s += ", " if parameters.first
        s += "size_t *block_number, size_t *block_size"
      end
      s += ")"
      s += ";\n" if final
      BOAST::get_output.print s if final
      return s
    end

    def call(*parameters)
      prefix = ""
      prefix += "call " if BOAST::get_lang==FORTRAN
      f = FuncCall::new(@name, *parameters)
      f.prefix = prefix
      return f
    end
    def decl(final=true)
      return self.decl_fortran(final) if BOAST::get_lang==FORTRAN
      return self.decl_c(final) if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def close(final=true)
      return self.close_fortran(final) if BOAST::get_lang==FORTRAN
      return self.close_c(final) if [C, CL, CUDA].include?( BOAST::get_lang )
    end
    def close_c(final=true)
      BOAST::decrement_indent_level
      s = ""
      s += "  return #{@properties[:return]};\n" if @properties[:return]
      s += "}"
      BOAST::get_output.puts s if final
      return s
    end
    def close_fortran(final=true)
      BOAST::decrement_indent_level
      s = ""
      if @properties[:return] then
        s += "  #{@name} = #{@properties[:return]}\n"
        s += "END FUNCTION #{@name}"
      else
        s += "END SUBROUTINE #{@name}"
      end
      BOAST::get_output.puts s if final
      return s
    end

    def print(final=true)
      s = self.decl
      if @block then
        @block.call
        s += self.close
      end
      return s
    end

    def decl_c(final=true)
      s = ""
#      s += self.header(BOAST::get_lang,false)
#      s += ";\n"
      if BOAST::get_lang == CL then
        if @properties[:local] then
          s += "static "
        else
          s += "__kernel "
          wgs = @properties[:reqd_work_group_size]
          if wgs then
            s += "__attribute__((reqd_work_group_size(#{wgs[0]},#{wgs[1]},#{wgs[2]}))) "
          end
        end
      elsif BOAST::get_lang == CUDA then
        if @properties[:local] then
          s += "static __device__ "
        else
          s += "__global__ "
          wgs = @properties[:reqd_work_group_size]
          if wgs then
            s += "__launch_bounds__(#{wgs[0]}*#{wgs[1]}*#{wgs[2]}) "
          end
        end
      end
      if @properties[:qualifiers] then
        s += "#{@properties[:qualifiers]} "
      end
      if @properties[:return] then
        s += "#{@properties[:return].type.decl} "
      else
        s += "void "
      end
      s += "#{@name}("
      if parameters.first then
        s += parameters.first.decl(false, @properties[:local])
        parameters[1..-1].each { |p|
          s += ", "+p.decl(false, @properties[:local])
        }
      end
      s += "){\n"
      BOAST::increment_indent_level
      constants.each { |c|
        s += " "*BOAST::get_indent_level
        s += c.decl(false)
        s += ";\n"
      }
      BOAST::get_output.print s if final
      return s
    end

    def decl_fortran(final=true)
      s = ""
      if @properties[:return] then
        s += "#{@properties[:return].type.decl} FUNCTION "
      else
        s += "SUBROUTINE "
      end
      s += "#{@name}("
      if parameters.first then
        s += parameters.first.to_s
        parameters[1..-1].each { |p|
          s += ", #{p}"
        }
      end
      s += ")\n"
      BOAST::increment_indent_level
      s += " "*BOAST::get_indent_level + "integer, parameter :: wp=kind(1.0d0)\n"
      constants.each { |c|
        s += " "*BOAST::get_indent_level
        s += c.decl(false)
        s += "\n"
      }
      parameters.each { |p|
        s += " "*BOAST::get_indent_level
        s += p.decl(false)
        s += "\n"
      }
      BOAST::get_output.print s if final
      return s
    end
  end

end
