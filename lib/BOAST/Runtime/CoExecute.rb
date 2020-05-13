module BOAST

  class CKernel

    module Synchro
      extend FFI::Library
      ffi_lib 'pthread'
      typedef :pointer, :mutex
      typedef :pointer, :cond
      typedef :pointer, :spin
      attach_function 'pthread_mutex_init', [ :mutex, :pointer ], :int
      attach_function 'pthread_mutex_destroy', [ :mutex ], :int
      #attach_function 'pthread_mutex_lock', [ :mutex ], :int
      #attach_function 'pthread_mutex_unlock', [ :mutex ], :int

      attach_function 'pthread_cond_init', [ :cond, :pointer ], :int
      attach_function 'pthread_cond_destroy', [ :cond ], :int
      #attach_function 'pthread_cond_wait', [ :cond, :mutex ], :int
      #attach_function 'pthread_cond_broadcast', [ :cond ], :int

      begin
        attach_function 'pthread_spin_init', [ :spin, :int ], :int
        attach_function 'pthread_spin_destroy', [ :spin ], :int
      rescue FFI::NotFoundError => e
        warn "spin functions not found"
      end
    end

    def self.coexecute(kernels)
      semaphore = Mutex.new
      pval = FFI::MemoryPointer::new(:int)
      pval.write_int(kernels.length)
      if synchro == 'MUTEX' then
        mutex = FFI::MemoryPointer::new(128)
        Synchro.pthread_mutex_init(mutex, nil)
        condition = FFI::MemoryPointer::new(128)
        Synchro.pthread_cond_init(mutex, nil)
        sync = [pval, mutex, condition]
      else
        spinlock = FFI::MemoryPointer::new(128)
        Synchro.pthread_spin_init(spinlock, 0)
        sync = [pval, spinlock]
      end
      threads = []
      returns = []
      args = []
      kernels.each_index { |i|
        kernels[i][0].build unless kernels[i][0].methods.include?(:run)
        args[i] = kernels[i][1].dup
        if args[i].last.kind_of?( Hash ) then
          args[i][-1] = args[i].last.dup
          args[i][-1][:coexecute] = sync
        else
          args[i].push( { :coexecute => sync } )
        end
      }
      kernels.each_index { |i|
        threads << Thread::new(i) { |j|
          ret = kernels[j][0].run(*args[j][0..-2],**args[j][-1])
          semaphore.synchronize {
            returns[j] = ret
          }
        }
      }
      threads.each { |thr| thr.join }
      if synchro == 'MUTEX' then
        Synchro.pthread_mutex_destroy(mutex)
        Synchro.pthread_cond_destroy(condition)
      else
        Synchro.pthread_spin_destroy(spinlock)
      end
      return returns
    end

  end

end
