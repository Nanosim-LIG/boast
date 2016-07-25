module BOAST

  # @!parse module Functors; functorize Procedure; end
  class Procedure
    include PrivateStateAccessor
    include Inspectable
    extend Functor
    include Annotation
    ANNOTATIONS = [ :name, :parameters, :constants ]

    attr_reader :name
    attr_reader :parameters
    attr_reader :constants
    attr_reader :properties
    attr_reader :headers

    # Creates a new Procedure
    # @param [#to_s] name Procedure identifier
    # @param [Array<Variable>] parameters list of the procedure parameters.
    # @param [Hash] properties set of named properties for the Procedure.
    # @option properties [Array<Variables>] :constants list of constant variables that are used in the Procedure. (see parameter in Fortran).
    # @option properties [Array<#to_s>] :headers list of headers that need to be included in order to compile the Procedure
    def initialize(name, parameters=[], properties={}, &block)
      @name = name
      @parameters = parameters
      @constants = properties[:constants]
      @constants = [] unless @constants
      @block = block
      @properties = properties
      @headers = properties[:headers]
      @headers = [] unless @headers
    end

    # @private
    def boast_header(lang=C)
      s = boast_header_s(lang)
      s += ";\n"
      output.print s
      return self
    end

    def call(*parameters)
      prefix = ""
      prefix += "call " if lang==FORTRAN
      f = FuncCall::new(@name, *parameters)
      f.prefix = prefix
      return f
    end

    def close
      return close_fortran if lang==FORTRAN
      return close_c if [C, CL, CUDA].include?( lang )
    end

    # Returns a {CKernel} with the Procedure as entry point.
    def ckernel
      old_output = output
      k = CKernel::new
      k.procedure = self
      self.pr
      set_output( old_output )
      return k
    end

    def pr
      open
      if @block then
        @block.call
        close
      end
      return self
    end

    def decl
      return decl_fortran if lang==FORTRAN
      return decl_c if [C, CL, CUDA].include?( lang )
    end

    def open
      return open_fortran if lang==FORTRAN
      return open_c if [C, CL, CUDA].include?( lang )
    end

    def to_s
      return decl_c_s if [C, CL, CUDA].include?( lang )
      return to_s_fortran if lang==FORTRAN
    end

    private

    def decl_fortran
      output.puts indent + "INTERFACE"
      increment_indent_level
      open_fortran
      close_fortran
      decrement_indent_level
      output.puts indent + "END INTERFACE"
      return self
    end

    def decl_c_s
      s = ""
      if lang == CL then
        if @properties[:local] then
          s += "#if __OPENCL_C_VERSION__ && __OPENCL_C_VERSION__ >= 120\n"
          s += "static\n"
          s += "#endif\n"
        else
          s += "__kernel "
          wgs = @properties[:reqd_work_group_size]
          if wgs then
            s += "__attribute__((reqd_work_group_size(#{wgs[0]},#{wgs[1]},#{wgs[2]}))) "
          end
        end
      elsif lang == CUDA then
        if @properties[:local] then
          s += "static __device__ "
        else
          s += "__global__ "
          wgs = @properties[:reqd_work_group_size]
          if wgs then
            s += "__launch_bounds__(#{wgs[0]}*#{wgs[1]}*#{wgs[2]}) "
          end
        end
      elsif lang == C then
        if @properties[:local] then
          s += "static "
        end
        if @properties[:inline] then
          s+= "inline "
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
        s += parameters.first.send(:decl_c_s, @properties[:local])
        parameters[1..-1].each { |p|
          s += ", "+p.send(:decl_c_s, @properties[:local])
        }
      end
      s += ")"
      return s
    end

    def decl_c
      s = indent + decl_c_s + ";"
      output.puts s
      return self
    end

    def to_s_fortran
      s = ""
      if @properties[:return] then
        s += "#{@properties[:return].type.decl} FUNCTION "
      else
        s += "SUBROUTINE "
      end
      s += "#{@name}("
      s += parameters.collect(&:name).join(", ")
      s += ")"
    end

    def open_c
      s = indent + decl_c_s + "{"
      output.puts s
      increment_indent_level
      constants.each { |c|
        BOAST::decl c
      }
      if lang == C then
        parameters.each { |p|
          align = p.align
          BOAST::pr align if align
        }
      end
      if @properties[:return] then
        BOAST::decl @properties[:return]
      end
      return self
    end

    def open_fortran
      s = indent + to_s_fortran
      s += "\n"
      increment_indent_level
      parameters.each { |p|
        p.type.define if p.type.kind_of? CStruct
      }
      s += indent + "integer, parameter :: wp=kind(1.0d0)"
      output.puts s
      constants.each { |c|
        BOAST::decl c
      }
      parameters.each { |p|
        BOAST::decl p
        align = p.align
        BOAST::pr align if align
      }
      if @properties[:return] then
        BOAST::decl @properties[:return]
      end
      return self
    end

    def close_c
      s = ""
      s += indent + "return #{@properties[:return]};\n" if @properties[:return]
      decrement_indent_level
      s += indent + "}"
      output.puts s
      return self
    end

    def close_fortran
      s = ""
      if @properties[:return] then
        s += indent + "#{@name} = #{@properties[:return]}\n"
        decrement_indent_level
        s += indent + "END FUNCTION #{@name}"
      else
        decrement_indent_level
        s += indent + "END SUBROUTINE #{@name}"
      end
      output.puts s
      return self
    end

    def boast_header_s( lang=C )
      s = ""
      headers.each { |h|
        s += "#include <#{h}>\n"
      }
      if lang == CL then
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
        s += parameters.first.boast_header(lang)
        parameters[1..-1].each { |p|
          s += ", "
          s += p.boast_header(lang)
        }
      end
      if lang == CUDA then
        s += ", " if parameters.first
        s += "size_t *block_number, size_t *block_size"
      end
      s += ")"
      return s
    end

  end

end
