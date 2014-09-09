module BOAST

  @@inspect = false

  def inspect?
    return @@inspect
  end

  def inspect=(val)
    @@inspect = val
  end

  extend self

  module Inspectable

    def inspect
      if BOAST::inspect? then
        variables = instance_variables.map{ |v|
          instance_variable_get(v) ? "#{v}=#{instance_variable_get(v).inspect}" : nil
        }.reject{ |v| v.nil? }.join(", ")
        "#<#{self.class}:#{(self.object_id<<1).to_s(16)}#{variables == "" ? "" : " #{variables}" }>" 
      else
        to_s
      end
    end

  end

end
