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

    def annotation(level)
      anns = {}
      self.class.const_get(:ANNOTATIONS).each { |a|
        var_sym = ("@" + a.to_s).to_sym
        var = self.instance_variable_get(var_sym)
        if var.is_a?(Annotation) and var.annotate_indepth?(level + 1) then
          anns[a] = { var.annotation_identifier => var.annotation(level + 1) }
        else
          anns[a] = var.to_s
        end
      }
      return anns
    end

  end

end
