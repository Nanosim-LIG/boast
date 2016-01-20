module BOAST

  module Annotation
    include PrivateStateAccessor

    def annotate_indepth?(level)
      return false if level > annotate_level
      return false unless annotate_indepth_list.include?(self.class.name.gsub("BOAST::", ""))
      return true
    end

    def annotation_identifier
      name = self.class.name.gsub("BOAST::", "")
      return "#{name}#{annotate_number(name)}"
    end

    def annotate_scalar(s, level)
      if s.is_a?(Annotation) and s.annotate_indepth?(level + 1) then
        return { s.annotation_identifier => s.annotation(level + 1) }
      elsif s.is_a?(Numeric)
        return s
      else
        return s.to_s
      end
    end

    def annotate_array(a, level)
      return a.collect { |e|
        annotate_var(e, level)
      }
    end

    def annotate_var(v, level)
      if v.is_a?(Array) then
        return annotate_array(v, level)
      else
        return annotate_scalar(v, level)
      end
    end

    def annotation(level)
      anns = {}
      self.class.const_get(:ANNOTATIONS).each { |a|
        var_sym = ("@" + a.to_s).to_sym
        var = self.instance_variable_get(var_sym)
        anns[a] = annotate_var(var, level)
      }
      return anns
    end

  end

end
