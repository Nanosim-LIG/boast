module BOAST
  @@inspect = false

  def BOAST.inspect
    return @@inspect
  end

  def BOAST.inspect=(val)
    @@inspect = val
  end

  module Inspectable

    def inspect
      if BOAST::inspect then
        variables = self.instance_variables.map{ |v|
          instance_variable_get(v) ? "#{v}=#{instance_variable_get(v).inspect}" : nil
        }.reject{ |v| v.nil? }.join(", ")
        "#<#{self.class}:#{(self.object_id<<1).to_s(16)}#{variables == "" ? "" : " #{variables}" }>" 
      else
        self.to_s
      end
    end

  end

end
