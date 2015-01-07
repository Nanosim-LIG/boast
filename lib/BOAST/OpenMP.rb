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

    def self.register_openmp_construct( name, c_strings, fortran_strings, open_clauses = [], end_clauses = [] )
      s = <<EOF
    class #{name} < OpenMPControlStructure

      def get_c_strings
        return { :begin => '#{c_strings[0]}',
                 :end => '#{c_strings[1]}' }
      end

      def get_fortran_strings
        return { :begin => '#{fortran_strings[0]}',
                 :end => '#{fortran_strings[1]}' }
      end

      def self.get_open_clauses
        return [ #{open_clauses.collect { |c| ":#{c}" }.join(",\n                 ")} ]
      end

      def self.get_end_clauses
        return [ #{end_clauses.collect { |c| ":#{c}" }.join(",\n                 ")} ]
      end

      def get_open_clauses
        return #{name}.get_open_clauses
      end

      def get_end_clauses
        return #{name}.get_end_clauses
      end

    end
EOF
      eval s
    end

    register_openmp_construct( :Parallel, [ '"#pragma omp parallel #{c}\n{"', '"}"' ],
                                          [ '"!$omp parallel #{c}"'         , '"!$omp end parallel"' ],
                                          [ :if,
                                            :num_threads,
                                            :default,
                                            :private,
                                            :firstprivate,
                                            :shared,
                                            :copyin,
                                            :reduction,
                                            :proc_bind ] )

    register_openmp_construct( :For, [ '"#pragma omp for #{c}"', '""' ],
                                     [ '"!$omp do #{c}"',        '"!$omp end do #{c}"' ],
                                     [ :private,
                                       :firstprivate,
                                       :lastprivate,
                                       :reduction,
                                       :schedule,
                                       :collapse,
                                       :ordered ],
                                     [ :nowait ] )

    register_openmp_construct( :Sections, [ '"#pragma omp sections #{c}\n{"', '"}"' ],
                                          [ '"!$omp sections #{c}"',          '"!$omp end sections #{c}"' ],
                                          [ :private,
                                            :firstprivate,
                                            :lastprivate,
                                            :reduction ],
                                          [ :nowait ] )

    register_openmp_construct( :Section, [ '"#pragma omp section\n{"', '"}"' ],
                                         [ '"!$omp section"',          '""' ] )

    register_openmp_construct( :Single, [ '"#pragma omp single #{c}\n{"', '"}"' ],
                                        [ '"!$omp single #{c}"',          '"!$omp end single #{c}"'],
                                        [ :private,
                                          :firstprivate ],
                                        [ :copyprivate,
                                          :nowait ] )

    register_openmp_construct( :Simd, [ '"#pragma omp simd #{c}"', '""' ],
                                      [ '"!$omp simd #{c}"',      '"!$omp end single"' ],
                                      [ :safelen,
                                        :linear,
                                        :aligned,
                                        :private,
                                        :lastprivate,
                                        :reduction,
                                        :collapse ] )

    register_openmp_construct( :DeclareSimd, [ '"#pragma omp declare simd #{c}"', '""' ],
                                             [ '"!$omp declare simd #{c}"',       '""' ],
                                             [ :simdlen,
                                               :linear,
                                               :aligned,
                                               :uniform,
                                               :inbranch,
                                               :notinbranch ] )
    register_openmp_construct( :ForSimd, [ '"#pragma omp for simd #{c}"', '""' ],
                                         [ '"!$omp do simd #{c}"',        '"!$omp end do simd #{c}"' ],
                                         (For.get_open_clauses + Simd.get_open_clauses).uniq,
                                         (For.get_end_clauses + Simd.get_end_clauses).uniq )

    register_openmp_construct( :TargetData, [ '"#pragma omp target data #{c}\n{"', '"}"' ],
                                            [ '"!$omp target data #{c}"',          '"!$omp end target data"' ],
                                            [ :device,
                                              :map,
                                              :if ] )

    register_openmp_construct( :Target, [ '"#pragma omp target #{c}\n{"', '"}"' ],
                                        [ '"!$omp target #{c}"',          '"!$omp end target"' ],
                                        [ :device,
                                          :map,
                                          :if ] )
    register_openmp_construct( :TargetUpdate, [ '"#pragma omp target update #{c}"', '""' ],
                                              [ '"!$omp target update #{c}"',       '""' ],
                                              [ :to,
                                                :from,
                                                :device,
                                                :if ] )
    register_openmp_construct( :DeclareTarget, [ '"#pragma omp declare target"', '"#pragma omp end declare target"' ],
                                               [ '"!$omp declare target ("',     '")"' ] )

    register_openmp_construct( :Teams, [ '"#pragma omp teams #{c}\n{"', '"}"' ],
                                       [ '"!$omp teams #{c}"',          '"!$omp end teams"' ],
                                       [ :num_teams,
                                         :thread_limit,
                                         :default,
                                         :private,
                                         :firstprivate,
                                         :shared,
                                         :reduction ] )

    register_openmp_construct( :Distribute, [ '"#pragma omp distribute #{c}"', '""' ],
                                            [ '"!$omp distribute"',            '"!$omp end distribute"' ],
                                            [ :private,
                                              :firstprivate,
                                              :collapse,
                                              :dist_schedule ] )

    register_openmp_construct( :DistributeSimd, [ '"#pragma omp distribute simd #{c}"', '""' ],
                                                [ '"!$omp distribute simd"',            '"!$omp end distribute simd"' ],
                                                (Distribute.get_open_clauses + Simd.get_open_clauses).uniq )

    register_openmp_construct( :DistributeParallelFor, [ '"#pragma omp distribute parallel for #{c}"', '""' ],
                                                       [ '"!$omp distribute parallel do #{c}"',        '"!$omp end distribute parallel do"' ],
                                                       (Distribute.get_open_clauses + Parallel.get_open_clauses + For.get_open_clauses).uniq )

    register_openmp_construct( :DistributeParallelForSimd, [ '"#pragma omp distribute parallel for simd #{c}"', '""' ],
                                                           [ '"!$omp distribute parallel do simd #{c}"',        '"!$omp end distribute parallel do simd"' ],
                                                           (Distribute.get_open_clauses + Parallel.get_open_clauses + For.get_open_clauses + Simd.get_open_clauses).uniq )

    register_openmp_construct( :ParallelFor, [ '"#pragma omp parallel for #{c}"', '""' ],
                                             [ '"!$omp parallel do #{c}"',        '"!$omp end parallel do"' ],
                                             (Parallel.get_open_clauses + For.get_open_clauses).uniq )

    register_openmp_construct( :ParallelSections, [ '"#pragma omp parallel sections #{c}\n{"', '"}"' ],
                                                  [ '"!$omp parallel sections #{c}"',          '"!$omp end parallel sections"' ],
                                                  (Parallel.get_open_clauses + For.get_open_clauses).uniq )

    register_openmp_construct( :ParallelForSimd, [ '"#pragma omp parallel for simd #{c}"', '""' ],
                                                 [ '"!$omp parallel do simd #{c}"',        '"!$omp end parallel do simd"' ],
                                                 (Parallel.get_open_clauses + For.get_open_clauses + Simd.get_open_clauses).uniq )

    register_openmp_construct( :TargetTeams, [ '"#pragma omp target teams #{c}\n{"', '"}"' ],
                                             [ '"!$omp target teams #{c}"',          '"!$omp end target teams"' ],
                                             (Target.get_open_clauses + Teams.get_open_clauses).uniq )

    register_openmp_construct( :TeamsDistribute, [ '"#pragma omp teams distribute #{c}"', '""' ],
                                                 [ '"!$omp teams distribute #{c}"',       '"!$omp end teams distribute"' ],
                                                 (Teams.get_open_clauses + Distribute.get_open_clauses).uniq )

    register_openmp_construct( :TeamsDistributeSimd, [ '"#pragma omp teams distribute simd #{c}"', '""' ],
                                                     [ '"!$omp teams distribute simd #{c}"',       '"!$omp end teams distribute simd"' ],
                                                     (Teams.get_open_clauses + Distribute.get_open_clauses + Simd.get_open_clauses).uniq )

    register_openmp_construct( :Task, [ '"#pragma omp task #{c}\n{"', '"}"' ],
                                      [ '"!$omp task #{c}"',          '"!$omp end task"' ],
                                      [ :if,
                                        :final,
                                        :untied,
                                        :default,
                                        :mergeable,
                                        :private,
                                        :firstprivate,
                                        :shared,
                                        :depend ] )

  end

end
