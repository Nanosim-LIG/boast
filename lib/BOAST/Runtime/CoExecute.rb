module BOAST

  class CKernel

    module Synchro
      extend FFI::Library
      ffi_lib FFI::Library::LIBC
      typedef :pointer, :mutex
      typedef :pointer, :cond
      attach_function 'pthread_mutex_init', [ :mutex, :pointer ], :int
      attach_function 'pthread_mutex_destroy', [ :mutex ], :int
      #attach_function 'pthread_mutex_lock', [ :mutex ], :int
      #attach_function 'pthread_mutex_unlock', [ :mutex ], :int

      attach_function 'pthread_cond_init', [ :cond, :pointer ], :int
      attach_function 'pthread_cond_destroy', [ :cond ], :int
      #attach_function 'pthread_cond_wait', [ :cond, :mutex ], :int
      #attach_function 'pthread_cond_broadcast', [ :cond ], :int
    end

    def self.coexecute(kernels)
      semaphore = Mutex.new
      pval = FFI::MemoryPointer::new(:int)
      pval.write_int(kernels.length)
      mutex = FFI::MemoryPointer::new(128)
      Synchro.pthread_mutex_init(mutex, nil)
      condition = FFI::MemoryPointer::new(128)
      Synchro.pthread_cond_init(mutex, nil)
      sync = [pval, mutex, condition]
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
          ret = kernels[j][0].run(*args[j])
          semaphore.synchronize {
            returns[j] = ret
          }
        }
      }
      threads.each { |thr| thr.join }
      Synchro.pthread_mutex_destroy(mutex)
      Synchro.pthread_cond_destroy(condition)
      return returns
    end

  end

end
