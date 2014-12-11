module BOAST

  module OpenMP

    module Pragma

      def openmp_pragma_to_s
        s = ""
        if lang == FORTRAN then
          s += "!$omp"
        elsif lang == C then
          s += "#pragma omp"
        else
          raise "Language does not support OpenMP!"
        end
        return s
      end

      def openmp_clauses_to_s
        s = ""
        if @openmp_clauses[:if] then
          s += " if(#{@openmp_clauses[:if]})"
        end
        if @openmp_clauses[:num_threads] then
          s += " num_threads(#{@openmp_clauses[:num_threads]})"
        end
        if @openmp_clauses[:default] then
          s += " default(#{@openmp_clauses[:default]})"
        end
        if @openmp_clauses[:private] then
          s += " private(#{[@openmp_clauses[:private]].flatten.join(", ")})"
        end
        if @openmp_clauses[:firstprivate] then
          s += " firstprivate(#{[@openmp_clauses[:firstprivate]].flatten.join(", ")})"
        end
        if @openmp_clauses[:lastprivate] then
          s += " lastprivate(#{[@openmp_clauses[:lastprivate]].flatten.join(", ")})"
        end
        if lang == C then
          if @openmp_clauses[:copyprivate] then
            s += " copyprivate(#{[@openmp_clauses[:copyprivate]].flatten.join(", ")})"
          end
          if @openmp_clauses[:nowait] then
            s += " nowait"
          end
        end
        if @openmp_clauses[:shared] then
          s += " shared(#{[@openmp_clauses[:shared]].flatten.join(", ")})"
        end
        if @openmp_clauses[:copyin] then
          s += " copyin(#{[@openmp_clauses[:copyin]].flatten.join(", ")})"
        end
        if @openmp_clauses[:reduction] then
          @openmp_clauses[:reduction].each { |identifier, list|
            s += " reduction(#{identifier}: #{list.join(", ")})"
          }
        end
        if @openmp_clauses[:depend] then
          @openmp_clauses[:depend].each { |type, list|
            s += " depend(#{type}: #{list.join(", ")})"
          }
        end
        if @openmp_clauses[:schedule] then
          s += " schedule(#{@openmp_clauses[:schedule].join(", ")})"
        end
        if @openmp_clauses[:collapse] then
          s += " collapse(#{@openmp_clauses[:collapse]})"
        end
        if @openmp_clauses[:ordered] then
          s += " ordered"
        end
        return s
      end

      def openmp_end_clauses_to_s
        s = ""
        if lang == FORTRAN then
          if @openmp_clauses[:copyprivate] then
            s += " copyprivate(#{[@openmp_clauses[:copyprivate]].flatten.join(", ")})"
          end
          if @openmp_clauses[:nowait] then
            s += " nowait"
          end
        end
        return s
      end

    end

    module_function
    def functorize(klass)
      name = klass.name.split('::').last
      s = <<EOF
    def #{name}(*args,&block)
       #{name}::new(*args,&block)
    end

    module_function :#{name}
EOF
      eval s
    end
 
    class OpenMPControlStructure < ControlStructure
      include OpenMP::Pragma

      def get_strings
        return { C => get_c_strings,
                 FORTRAN => get_fortran_strings }
      end

      def initialize(options = {}, &block)
        @openmp_clauses = options
        @block = block
      end

      eval token_string_generator( * %w{begin c})
      eval token_string_generator( * %w{end c})

      def to_s
        return begin_string(openmp_clauses_to_s)
      end

      def open
        output.puts to_s
        return self
      end

      def pr(*args)
        open
        if @block then
          @block.call(*args)
          close
        end
        return self
      end

      def close
        output.puts end_string(openmp_end_clauses_to_s) 
        return self
      end
     
    end
 
    class Parallel < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp parallel #{c}\n{"',
                 :end => '"}"' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp parallel #{c}"',
                 :end => '"!$omp end parallel #{c}"' }
      end

    end

    class For < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp for #{c}"',
                 :end => '""' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp do #{c}"',
                 :end => '"!$omp end do #{c}"' }
      end

    end

  end

end
