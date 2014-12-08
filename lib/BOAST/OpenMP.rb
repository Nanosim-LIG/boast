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
          options[:reduction].each { |identifier, list|
            s += " reduction(#{identifier}: #{list.join(", ")})"
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
  
    def var_functorize(klass)
      name = klass.name.split('::').last
      s = <<EOF
    def #{name}(*args,&block)
       Variable::new(args[0],#{name},*args[1..-1], &block)
    end

    module_function :#{name}
EOF
      eval s
    end

    module Functor

      def self.extended(mod)
        BOAST::OpenMP::functorize(mod)
      end

    end

    class ControlStructure
      include PrivateStateAccessor
      include Inspectable
      include OpenMP::Pragma

      def self.inherited(child)
        child.extend Functor
      end

      def self.token_string_generator(name, *args)
        s = <<EOF
      def #{name}_string(#{args.join(",")})
        return eval @@strings[get_lang][:#{name}] 
      end
EOF
      end

    end

    class Parallel < ControlStructure

      def initialize(options = {}, &block)
        @openmp_clauses = options
        @block = block
      end
      @@c_strings = {
        :parallel => '"#pragma omp parallel #{c}\n{"',
        :end => '"}"',
      }

      @@f_strings = {
        :parallel => '"!$omp parallel #{c}"',
        :end => '"!$omp end parallel #{c}"',
      }

      @@strings = {
        C => @@c_strings,
        FORTRAN => @@f_strings
      }

      eval token_string_generator( * %w{parallel c})
      eval token_string_generator( * %w{end c})

      def to_s
        return parallel_string(openmp_clauses_to_s)
      end

      def open
        output.puts to_s
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
      end

    end

  end

end
