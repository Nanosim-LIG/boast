module BOAST

  class CKernel

    def self.coexecute(kernels)
      semaphore = Mutex.new
      threads = []
      returns = []
      args = []
      kernels.each_index { |i|
        kernels[i][0].build unless kernels[i][0].methods.include?(:run)
        args[i] = kernels[i][1].dup
        if args[i].last.kind_of?( Hash ) then
          args[i][-1] = args.last.dup
          args[i][-1][:coexecute] = true
        else
          args[i].push( { :coexecute => true } )
        end
      }
      kernels.each_index { |i|
        threads << Thread::new(i) { |j|
          ret = kernels[j][0].run(*args[j])
          semaphore.synchronize {
            returns[j] = ret
          }
        }
      }
      threads.each { |thr| thr.join }
      return returns
    end

  end

end
