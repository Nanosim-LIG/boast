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

      # Registers an openmp clause, arg_type can be :none, :option, :option_list, :simple, :list, :multilist
      def self.register_clause( name, arg_type )
        s = <<EOF
      def openmp_clause_#{name}(c)
EOF
        case arg_type
        when :none
          s += <<EOF
        return " #{name}"
EOF
        when :option
          s += <<EOF
        return " \#{c}"
EOF
        when :option_list
          s += <<EOF
        return " (\#{[c].flatten.join(", ")})"
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
      register_clause(:seq_cst,      :none)
      register_clause(:name,         :option)
      register_clause(:flush_list,   :option_list)
      register_clause(:threadprivate_list,   :option_list)
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

    def self.register_openmp_construct( name, c_name, open_clauses, end_clauses = [], options = {} )
      fortran_name = c_name
      fortran_name = options[:fortran_name] if options[:fortran_name]
      s = <<EOF
    class #{name} < OpenMPControlStructure

      def self.name
        return "#{name}"
      end

      def self.c_name
        return "#{c_name}"
      end

      def self.fortran_name
        return "#{fortran_name}"
      end

      def self.block
        return #{options[:block] ? "true" : "false"}
      end

      def get_c_strings
        return { :begin => '"#pragma omp #{c_name} #{ open_clauses.length + end_clauses.length > 0 ? "\#{c}" : "" }#{ options[:block] ? "\\n{" : "" }"',
EOF
      if options[:c_end] then
        s += <<EOF
                 :end => '"#pragma omp end #{c_name}"' }
EOF
      else
        s += <<EOF
                 :end => '"#{ options[:block] ? "}" : "" }"' }
EOF
      end
        s += <<EOF
      end

      def get_fortran_strings
        return { :begin => '"!$omp #{fortran_name}#{ open_clauses.length > 0 ? " \#{c}" : "" }#{ options[:fortran_block] ? "(" : "" }"',
EOF
      if options[:fortran_no_end] then
        s += <<EOF
                 :end => '"#{ options[:fortran_block] ? ")" : "" }"' }
EOF
      else
        s += <<EOF
                 :end => '"!$omp end #{fortran_name} #{ end_clauses.length > 0 ? "\#{c}" : "" }"' }
EOF
      end
        s += <<EOF
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

    def self.register_openmp_compound_construct( c1, c2, options = {} )
      register_openmp_construct( c1.name+c2.name, "#{c1.c_name} #{c2.c_name}",
                                                  (c1.get_open_clauses + c2.get_open_clauses).uniq,
                                                  options[:no_end_clauses] ? [] : (c1.get_end_clauses + c2.get_end_clauses).uniq,
                                                  :fortran_name => "#{c1.fortran_name} #{c2.fortran_name}",
                                                  :block => options[:block] )
    end

    register_openmp_construct( :Parallel, "parallel",
                                          [ :if,
                                            :num_threads,
                                            :default,
                                            :private,
                                            :firstprivate,
                                            :shared,
                                            :copyin,
                                            :reduction,
                                            :proc_bind ],
                                          [],
                                          :block => true )

    register_openmp_construct( :For, "for",
                                     [ :private,
                                       :firstprivate,
                                       :lastprivate,
                                       :reduction,
                                       :schedule,
                                       :collapse,
                                       :ordered ],
                                     [ :nowait ],
                                     :fortran_name => "do" )

    register_openmp_construct( :Sections, "sections",
                                          [ :private,
                                            :firstprivate,
                                            :lastprivate,
                                            :reduction ],
                                          [ :nowait ],
                                          :block => true )

    register_openmp_construct( :Section, "section", [], [],
                                         :block => true,
                                         :fortran_no_end => true )

    register_openmp_construct( :Single, "single",
                                        [ :private,
                                          :firstprivate ],
                                        [ :copyprivate,
                                          :nowait ],
                                        :block => true )

    register_openmp_construct( :Simd, "simd",
                                      [ :safelen,
                                        :linear,
                                        :aligned,
                                        :private,
                                        :lastprivate,
                                        :reduction,
                                        :collapse ] )

    register_openmp_construct( :DeclareSimd, "declare simd",
                                             [ :simdlen,
                                               :linear,
                                               :aligned,
                                               :uniform,
                                               :inbranch,
                                               :notinbranch ] )

    register_openmp_compound_construct( For, Simd )

    register_openmp_construct( :TargetData, "target data",
                                            [ :device,
                                              :map,
                                              :if ],
                                            [],
                                            :block => true )

    register_openmp_construct( :Target, "target",
                                        [ :device,
                                          :map,
                                          :if ],
                                        [],
                                        :block => true )

    register_openmp_construct( :TargetUpdate, "target update",
                                              [ :to,
                                                :from,
                                                :device,
                                                :if ],
                                              [],
                                              :fortran_no_end => true )

    register_openmp_construct( :DeclareTarget, "declare target",
                                               [],
                                               [],
                                               :c_end => true,
                                               :fortran_no_end => true,
                                               :fortran_block => true )

    register_openmp_construct( :Teams, "teams",
                                       [ :num_teams,
                                         :thread_limit,
                                         :default,
                                         :private,
                                         :firstprivate,
                                         :shared,
                                         :reduction ],
                                       [],
                                       :block => true )

    register_openmp_construct( :Distribute, "distribute",
                                            [ :private,
                                              :firstprivate,
                                              :collapse,
                                              :dist_schedule ] )

    register_openmp_compound_construct( Distribute, Simd )

    register_openmp_compound_construct( Parallel, For, :no_end_clauses => true )

    register_openmp_compound_construct( Distribute, ParallelFor )

    register_openmp_compound_construct( Parallel, ForSimd, :no_end_clauses => true )

    register_openmp_compound_construct( Distribute, ParallelForSimd )

    register_openmp_compound_construct( Parallel, Sections, :no_end_clauses => true, :block => true )

    register_openmp_compound_construct( Target, Teams, :block => true )

    register_openmp_compound_construct( Teams, Distribute )

    register_openmp_compound_construct( Teams, DistributeSimd )

    register_openmp_compound_construct( Target, TeamsDistribute )

    register_openmp_compound_construct( Target, TeamsDistributeSimd )

    register_openmp_compound_construct( Teams, DistributeParallelFor )

    register_openmp_compound_construct( Target, TeamsDistributeParallelFor )

    register_openmp_compound_construct( Teams, DistributeParallelForSimd )

    register_openmp_compound_construct( Target, TeamsDistributeParallelForSimd )

    register_openmp_construct( :Task, "task",
                                      [ :if,
                                        :final,
                                        :untied,
                                        :default,
                                        :mergeable,
                                        :private,
                                        :firstprivate,
                                        :shared,
                                        :depend ],
                                      [],
                                      :block => true )

    register_openmp_construct( :Taskyield, "taskyield", [], [], :fortran_no_end => true )

    register_openmp_construct( :Master, "master", [], [], :block => true )

    register_openmp_construct( :Critical, "critical", [:name], [:name], :block => true )

    register_openmp_construct( :Barrier, "barrier", [], [], :fortran_no_end => true )

    register_openmp_construct( :Taskwait, "taskwait", [], [], :fortran_no_end => true )

    register_openmp_construct( :Taskgroup, "taskgroup", [], [], :block => true )

    register_openmp_construct( :AtomicRead, "atomic read", [:seq_cst], [] )

    register_openmp_construct( :AtomicWrite, "atomic write", [:seq_cst], [] )

    register_openmp_construct( :AtomicUpdate, "atomic update", [:seq_cst], [] )

    register_openmp_construct( :AtomicCapture, "atomic capture", [:seq_cst], [], :block => true )

    register_openmp_construct( :Flush, "flush", [:flush_list], [], :fortran_no_end => true )

    register_openmp_construct( :Ordered, "ordered", [], [], :block => true )

    register_openmp_construct( :CancelParallel, "cancel parallel", [:if], [], :fortran_no_end => true )

    register_openmp_construct( :CancelSections, "cancel sections", [:if], [], :fortran_no_end => true )

    register_openmp_construct( :CancelFor, "cancel for", [:if], [], :fortran_no_end => true )

    register_openmp_construct( :CancelTaskgroup, "cancel taskgroup", [:if], [], :fortran_no_end => true )

    register_openmp_construct( :CancellationPointParallel, "cancellation point parallel", [], [], :fortran_no_end => true )

    register_openmp_construct( :CancellationPointSections, "cancellation point sections", [], [], :fortran_no_end => true )

    register_openmp_construct( :CancellationPointFor, "cancellation point for", [], [], :fortran_no_end => true )

    register_openmp_construct( :CancellationPointTaskgroup, "cancellation point taskgroup", [], [], :fortran_no_end => true )

    register_openmp_construct( :Threadprivate, "threadprivate", [:threadprivate_list], [], :fortran_no_end => true )

  end

end
