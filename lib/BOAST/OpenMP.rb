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

      # Registers an openmp clause, arg_type can be :none, :simple, :list, :multilist
      def self.register_clause( name, arg_type )
        s = <<EOF
      def openmp_clause_#{name}(c)
EOF
        case arg_type
        when :none
          s += <<EOF
        return " #{name}"
EOF
        when :simple
          s += <<EOF
        return " #{name}(\#{c})"
EOF
        when :list
s += <<EOF
        return " #{name}(\#{[c].flatten.join(", ")})"
EOF
        when :multilist
s += <<EOF
        s = ""
        c.each { |id, list|
          s += " #{name}(\#{id}: \#{[list].flatten.join(", ")})"
        }
        return s
EOF
        else
          raise "Unknown argument type!"
        end
        s += <<EOF
      end
EOF
        eval s
      end

      register_clause(:nowait,       :none)
      register_clause(:ordered,      :none)
      register_clause(:untied,       :none)
      register_clause(:mergeable,    :none)
      register_clause(:inbranch,     :none)
      register_clause(:notinbranch,  :none)
      register_clause(:if,           :simple)
      register_clause(:num_threads,  :simple)
      register_clause(:default,      :simple)
      register_clause(:collapse,     :simple)
      register_clause(:safelen,      :simple)
      register_clause(:simdlen,      :simple)
      register_clause(:device,       :simple)
      register_clause(:private,      :list)
      register_clause(:shared,       :list)
      register_clause(:firstprivate, :list)
      register_clause(:lastprivate,  :list)
      register_clause(:copyprivate,  :list)
      register_clause(:copyin,       :list)
      register_clause(:schedule,     :list)
      register_clause(:linear,       :list)
      register_clause(:aligned,      :list)
      register_clause(:uniform,      :list)
      register_clause(:to,           :list)
      register_clause(:from,         :list)
      register_clause(:reduction,    :multilist)
      register_clause(:depend,       :multilist)
      register_clause(:map,          :multilist)

      def openmp_open_clauses_to_s
        s = ""
        get_open_clauses.each { |c|
          s += self.send( "openmp_clause_#{c}", @openmp_clauses[c] ) if @openmp_clauses[c]
        }
        if lang == C then
          get_end_clauses.each { |c|
            s += self.send( c, @openmp_clauses[c] ) if @openmp_clauses[c]
          }
        end
        return s
      end

      def openmp_end_clauses_to_s
        s = ""
        if lang == FORTRAN then
          get_end_clauses.each { |c|
            s += self.send( c, @openmp_clauses[c] ) if @openmp_clauses[c]
          }
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
        return begin_string(openmp_open_clauses_to_s)
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
                 :end => '"!$omp end parallel"' }
      end

      def self.get_open_clauses
        return [ :if,
                 :num_threads,
                 :default,
                 :private,
                 :firstprivate,
                 :shared,
                 :copyin,
                 :reduction,
                 :proc_bind ]
      end

      def get_open_clauses
        return Parallel.get_open_clauses
      end

      def get_end_clauses
        return [ ]
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

      def self.get_open_clauses
        return [ :private,
                 :firstprivate,
                 :lastprivate,
                 :reduction,
                 :schedule,
                 :collapse,
                 :ordered ]
      end

      def get_open_clauses
        return For.get_open_clauses
      end

      def self.get_end_clauses
        return [ :nowait ]
      end

      def get_end_clauses
        return For.get_end_clauses
      end

    end

    class Sections < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp sections #{c}\n{"',
                 :end => '"}"' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp sections #{c}"',
                 :end => '"!$omp end sections #{c}"' }
      end

      def get_open_clauses
        return [ :private,
                 :firstprivate,
                 :lastprivate,
                 :reduction ]
      end

      def get_end_clauses
        return [ :nowait ]
      end

    end

    class Section < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp section\n{"',
                 :end => '"}"' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp section"',
                 :end => '""' }
      end

      def get_open_clauses
        return [ ]
      end

      def get_end_clauses
        return [ ]
      end

    end

    class Single < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp single #{c}\n{"',
                 :end => '"}"' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp single #{c}"',
                 :end => '"!$omp end single #{c}"' }
      end

      def get_open_clauses
        return [ :private,
                 :firstprivate ]
      end

      def get_end_clauses
        return [ :copyprivate,
                 :nowait ]
      end

    end

    class Simd < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp simd #{c}"',
                 :end => '""' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp simd #{c}"',
                 :end => '"!$omp end single"' }
      end

      def self.get_open_clauses
        return [ :safelen,
                 :linear,
                 :aligned,
                 :private,
                 :lastprivate,
                 :reduction,
                 :collapse ]
      end

      def get_open_clauses
        return Simd.get_open_clauses
      end

      def self.get_end_clauses
        return [ ]
      end

      def get_end_clauses
        return Simd.get_end_clauses
      end

    end

    class DeclareSimd < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp declare simd #{c}"',
                 :end => '""' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp declare simd #{c}"',
                 :end => '""' }
      end

      def get_open_clauses
        return [ :simdlen,
                 :linear,
                 :aligned,
                 :uniform,
                 :inbranch,
                 :notinbranch ]
      end

      def get_end_clauses
        return [ ]
      end

    end

    class ForSimd < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp for simd #{c}"',
                 :end => '""' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp do simd #{c}"',
                 :end => '"!$omp end do simd #{c}"' }
      end

      def get_open_clauses
        return (For.get_open_clauses + Simd.get_open_clauses).uniq
      end

      def get_end_clauses
        return (For.get_end_clauses + Simd.get_end_clauses).uniq
      end

    end

    class TargetData < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp target data #{c}\n{"',
                 :end => '"}"' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp target data #{c}"',
                 :end => '"!$omp end target data"' }
      end

      def get_open_clauses
        return [ :device,
                 :map,
                 :if ]
      end

      def get_end_clauses
        return [ ]
      end

    end

    class Target < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp target #{c}\n{"',
                 :end => '"}"' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp target #{c}"',
                 :end => '"!$omp end target"' }
      end

      def get_open_clauses
        return [ :device,
                 :map,
                 :if ]
      end

      def get_end_clauses
        return [ ]
      end

    end

    class TargetUpdate < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp target update #{c}"',
                 :end => '""' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp target update #{c}"',
                 :end => '""' }
      end

      def get_open_clauses
        return [ :to,
                 :from,
                 :device,
                 :if ]
      end

      def get_end_clauses
        return [ ]
      end

    end

    class DeclareTarget < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp declare target"',
                 :end => '"#pragma omp end declare target"' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp declare target"',
                 :end => '""' }
      end

      def get_open_clauses
        return [ :to,
                 :from,
                 :device,
                 :if ]
      end

      def get_end_clauses
        return [ ]
      end

    end

    class ParallelFor < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp parallel for #{c}"',
                 :end => '""' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp parallel do #{c}"',
                 :end => '"!$omp end parallel do"' }
      end

      def get_open_clauses
        return (For.get_open_clauses + Parallel.get_open_clauses).uniq
      end

      def get_end_clauses
        return [ ]
      end

    end

    class Task < OpenMPControlStructure

      def get_c_strings
        return { :begin => '"#pragma omp task #{c}\n{"',
                 :end => '"}"' }
      end

      def get_fortran_strings
        return { :begin => '"!$omp task #{c}"',
                 :end => '"!$omp end task"' }
      end

      def get_open_clauses
        return [ :if,
                 :final,
                 :untied,
                 :default,
                 :mergeable,
                 :private,
                 :firstprivate,
                 :shared,
                 :depend ]
      end

      def get_end_clauses
        return [ ]
      end

    end

  end

end
