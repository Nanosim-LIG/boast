module BOAST
  module TypeTransition
    @@transitions = Hash::new { |hash, key| hash[key] = Hash::new{ |has, ke| has[ke] = Hash::new } }
    def get_transition(type1, type2, operator)
      #STDERR.puts @@transitions.inspect
      match_type1 = @@transitions.keys.select { |e1| true if type1 <= e1 }
      raise "Unknown type!" if match_type1.length == 0
      match_type1.sort!
      #STDERR.puts match_type1.inspect
      match_type1.each { |t1|
        match_type2 = @@transitions[t1].keys.select{ |e2| true if type2 <= e2 }
        match_type2.sort!
        #STDERR.puts match_type2.inspect
        match_type2.each { |t2|
          #STDERR.puts @@transitions[t1][t2].inspect
          return [@@transitions[t1][t2][operator], operator] if @@transitions[t1][t2][operator]
          return [@@transitions[t1][t2][:default], operator] if @@transitions[t1][t2][:default]
        }
      }
      raise "Unresolvable transition!"
    end

    def set_transition(type1, type2, operator, return_type)
      @@transitions[type1][type2][operator] = return_type
    end

    def transition(var1, var2, operator)
      signed = false
      size = nil
      return_type, operator = get_transition(var1.type.class, var2.type.class, operator)
      #STDERR.puts "#{return_type} : #{var1.type.class} #{operator} #{var2.type.class}"
      if var1.type.class <= return_type and var2.type.class <= return_type then
        signed = signed or var1.type.signed if var1.type.respond_to?(:signed)
        signed = signed or var2.type.signed if var2.type.respond_to?(:signed)
        if var1.type.respond_to?(:size) and var2.type.respond_to?(:size) then
          size = [var1.type.size, var2.type.size].max
        end
        [BOAST::Variable::new("dummy", return_type, :size => size, :signed => signed), operator]
      elsif var1.type.class <= return_type then
        return [var1, operator]
      elsif var2.type.class <= return_type then
        return [var2, operator]
      end
    end
  end

end
