module BOAST
  module Temperature
    extend PrivateStateAccessor
    module_function

    def get
      r = {}
      Dir.glob("/sys/devices/platform/coretemp.*") {|dir|
        cpu = dir.match(/[0-9]*$/)[0].to_i
        r[cpu] = {}
        Dir.glob(dir+'/hwmon/hwmon*/temp*_input') {|fname|
          sensor = fname.match(/temp([0-9]+)_input$/)[1].to_i - 1
          f = File.open(fname, 'r')
          r[cpu][sensor] = f.read.to_i
          f.close
        }
      }
      return r
    end

    def set min_temps=nil
      return if nil==min_temps
      now = self.get
      now = [now[0][0],now[1][0]]
      fin = true
      (0...now.length).each {|i|
        fin = false if now[i] < min_temps[i]
      }
      return if fin
      p = Array.new
      (0...24).each{|x| p.push fork{self.busy}}
      while true
        now = self.get
        now = [now[0][0],now[1][0]]
        fin = true
        (0...now.length).each {|i|
          fin = false if now[i] < min_temps[i]
        }
        break if fin
      end
      p.each {|x| Process.kill "KILL",x}
      p.each {|x|
        begin
          Process.wait x
        rescue Errno::ECHILD
        end
      }
    end

    def busy
      while true
        4.2 / 2.78
      end
    end

  end
end
