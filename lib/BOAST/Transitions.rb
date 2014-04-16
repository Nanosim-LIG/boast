module BOAST
  module TypeTransition
    @@transitions = Hash::new { |hash, key| hash[key] = Hash::new }
    def get_transition(type1, type2, operator)
      #STDERR.puts @@transitions.inspect
      #STDERR.puts @@transitions[t1][t2].inspec
      ops = @@transitions[[type1,type2]]
      raise "Types #{[type1,type2]} have no relation!" if not ops
      t = ops[operator]
      return [t,operator] if t
      t = ops[:default]
      return [t,operator] if t
      raise "Unresolvable transition!"
     end

     def set_transition(type1, type2, operator, return_type)
       @@transitions[[type1,type2]][operator] = return_type
     end

     def transition(var1, var2, operator)
       signed = false
       size = nil
       vector_length = 1
       return_type, operator = get_transition(var1.type.class, var2.type.class, operator)
       #STDERR.puts "#{return_type} : #{var1.type.class} #{operator} #{var2.type.class}"
       if var1.type.class == return_type and var2.type.class == return_type then
         signed = signed or var1.type.signed
         signed = signed or var2.type.signed
         size = [var1.type.size, var2.type.size].max
         vector_length = [var1.type.vector_length, var2.type.vector_length].max
         [BOAST::Variable::new("dummy", return_type, :size => size, :signed => signed), operator]
       elsif var1.type.class == return_type then
         return [var1, operator]
       elsif var2.type.class == return_type then
         return [var2, operator]
       end
     end
  end
end
