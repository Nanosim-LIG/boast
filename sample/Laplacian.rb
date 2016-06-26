[ '../lib', 'lib' '.' ].each { |d| $:.unshift(d) if File::directory?(d) }
require 'BOAST'
include BOAST
require 'narray'
require 'opencl_ruby_ffi'

class Numeric
  def clamp( min, max )
    [[self, max].min, min].max
  end
end

def rndup( val, div)
  return (val%div) == 0 ? val : val + div - (val%div)
end

set_array_start(0)
register_funccall("clamp")

def laplacian_ref( input, output, width, height )
  (1...(height-1)).each { |y_i|
    (3...(3*width-3)).each { |x_i|
      output[x_i, y_i] = ( -input[x_i - 3, y_i - 1] - input[x_i, y_i - 1] - input[x_i + 3, y_i - 1]\
                          - input[x_i - 3, y_i]     + input[x_i, y_i] * 9 - input[x_i + 3, y_i]\
                          - input[x_i - 3, y_i + 1] - input[x_i, y_i + 1] - input[x_i + 3, y_i + 1] ).clamp(0,255)
    }
  }
end

def laplacian_c_ref
  k = CKernel::new
  height = Int("height", :dir => :in)
  width = Int("width", :dir => :in)
  w = Int("w")
  pdst = Int("pdst", :dir => :out,  :signed => false, :size => 1, :dim => [ Dim(3), Dim(width), Dim(height) ] )
  psrc = Int("psrc", :dir => :in, :signed => false, :size => 1, :dim => [ Dim(3), Dim(width), Dim(height) ] )
  p = Procedure("math", [width, height, psrc, pdst]) {
    decl i = Int("i")
    decl j = Int("j")
    decl c = Int("c")
    decl tmp = Int("tmp")
    decl w
    pr w === width * 3
    pr For(j, 1, height-2) {
      pr For(i, 1, width-2) {
        pr For(c, 0, 2) {
          pr tmp === (  -psrc[c, i-1, j-1] - psrc[c, i, j-1] -   psrc[c, i+1, j-1]\
                       - psrc[c, i-1, j]   + psrc[c, i, j] * 9 - psrc[c, i+1, j]\
                       - psrc[c, i-1, j+1] - psrc[c, i, j+1] -   psrc[c, i+1, j+1] )
          pr pdst[c,i,j] === Ternary(tmp < 0, 0, Ternary(tmp>255, 255, tmp))
        }
      }
    }
  }
  pr p
  k.procedure = p
  return k
end

def split_in_ranges(length, base_split_length)
  ranges = []
  temp_split_length = base_split_length
  temp_length = length
  begin
    if temp_split_length <= temp_length then
      first = length - temp_length
      last = first + temp_split_length-1
      ranges.push (first..last)
      temp_length -= temp_split_length
    else
      temp_split_length /= 2
    end
  end while temp_length > 0
  return ranges
end

def find_in_ranges(ranges, start_indx, end_indx)
  vec_indx = ranges.find_index { |r| r.include?(start_indx) }
  delta = 0
  range = nil
  if vec_indx then
    r = ranges[vec_indx]
    range = (start_indx-r.begin)..(end_indx < r.end ? end_indx - r.begin : r.end - r.begin)
    #beware valid component idexing length are 1,2,3,4,8,16
    if not [1, 2, 3, 4, 8, 16].include?(range.end - range.begin + 1) then
      new_size = [1, 2, 3, 4, 8, 16].rindex { |e| e < (range.end - range.begin + 1) }
      range = range.begin..(range.begin+new_size-1)
    end
    delta = range.end - range.begin + 1
  end

  return [vec_indx, range, delta]
end

def merge_vectors(vectors, ranges, start_indx, end_indx)
  merge_expr = []
  begin
    vec_indx, range, delta = find_in_ranges(ranges, start_indx, end_indx)
    if vec_indx then
      vec = vectors[vec_indx]
      merge_expr.push( "#{vec.components(range)}" )
      start_indx += delta
    else # in the case vectors have dummy elements...
      (end_indx - start_indx + 1).times {
        merge_expr.push( "0" )
      }
      start_indx = end_indx + 1
    end
  end while start_indx <= end_indx
  return merge_expr
end

def laplacian(options)

  default_options = {:x_component_number => 1, :vector_length => 1, :y_component_number => 1, :temporary_size => 2, :vector_recompute => false, :load_overlap => false}
  opts = default_options.update(options)
  x_component_number = opts[:x_component_number]
  vector_length      = opts[:vector_length]
  y_component_number = opts[:y_component_number]
  temporary_size     = opts[:temporary_size]
  vector_recompute   = opts[:vector_recompute]
  load_overlap       = opts[:load_overlap]
 
  k = CKernel::new
  height = Int("height", :dir => :in)
  width = Int("width", :dir => :in)
  w = Int("w")
  pdst = Int("pdst", :dir => :out,  :signed => false, :size => 1, :dim => [ Dim(w), Dim(height) ] )
  psrc = Int("psrc", :dir => :in, :signed => false, :size => 1, :dim => [ Dim(w), Dim(height) ] )
  
  p = Procedure("math", [psrc, pdst, width, height]) {
    decl y = Int("y")
    decl x = Int("x")
    decl w
    
    pr x === get_global_id(0) * x_component_number 
    pr y === get_global_id(1) * y_component_number
    pr w === width * 3

    vector_number = (x_component_number.to_f/vector_length).ceil
    total_x_size = vector_recompute ? vector_number * vector_length : x_component_number

    x_offset = total_x_size + 3
    y_offset = y_component_number + 1

    pr x === Ternary(x < 3, 3, Ternary( x > w      - x_offset, w      - x_offset, x ) )
    pr y === Ternary(y < 1, 1, Ternary( y > height - y_offset, height - y_offset, y ) )

    temp_type = "#{Int("dummy", :size => temporary_size).type.decl}"
    temp_vec_type = "#{Int("dummy", :size => temporary_size, :vector_length => vector_length).type.decl}"
    out_type  = "#{Int("dummy", :size => 1, :signed => false).type.decl}"
    out_vec_type = "#{Int("dummy", :size => 1, :signed => false, :vector_length => vector_length).type.decl}"

    if not load_overlap then
      total_load_window = total_x_size + 6
      tempload = []
      ranges = split_in_ranges(total_load_window, vector_length)
      ranges.each { |r|
          tempload.push( Int("tempload#{r.begin}_#{r.end}", :size => 1, :vector_length => (r.end - r.begin + 1), :signed => false) )
      }
      decl *(tempload)
    else
      tempnn = (0..2).collect { |v_i|
        (0...vector_number).collect { |x_i|
          (0...(y_component_number+2)).collect { |y_i|
            Int("temp#{x_i}#{v_i}#{y_i}", :size => 1, :vector_length => vector_length, :signed => false)
          }
        }
      }
      decl *(tempnn.flatten)
    end
    resnn = (0...(vector_number)).collect { |v_i|
      (0...(y_component_number)).collect { |y_i|
        Int("res#{v_i}#{y_i}", :size => 1, :vector_length => vector_length, :signed => false)
      }
    }
    decl *(resnn.flatten)

    tempcnn = (0..2).collect { |v_i|
      (0...vector_number).collect { |x_i|
        (0...(y_component_number+2)).collect { |y_i|
          Int("tempc#{x_i}#{v_i}#{y_i}", :size => temporary_size, :vector_length => vector_length)
        }
      }
    }
    decl *(tempcnn.flatten)
    rescnn = (0...vector_number).collect { |v_i|
      (0...y_component_number).collect { |y_i|
        Int("resc#{v_i}#{y_i}", :size => temporary_size, :vector_length => vector_length)
      }
    }
    decl *(rescnn.flatten)

    (0...(y_component_number+2)).each { |y_i|
      if not load_overlap then
        load_start = -3
        tempload.each{ |v|
          pr v === psrc[x + load_start, y + (y_i - 1)]
          load_start += v.type.vector_length
        }

        (0..2).each { |x_i|
          (0...vector_number).each { |v_i|
            start_indx = v_i * vector_length + x_i * 3
            end_indx = start_indx + vector_length - 1
            merge_expr = merge_vectors(tempload, ranges, start_indx, end_indx)
            pr tempcnn[x_i][v_i][y_i] === Int( "(#{out_vec_type})(#{merge_expr.join(",")})", :size => 1, :vector_length => vector_length)
          }
        }
      else
        (0..2).each { |x_i|
          (0...vector_number).each { |v_i|
            pr tempnn[x_i][v_i][y_i] === psrc[x + v_i * vector_length + 3 * (x_i - 1), y + (y_i - 1)]
            pr tempcnn[x_i][v_i][y_i] === tempnn[x_i][v_i][y_i]
          }
        }
      end
    }
    (0...vector_number).each { |v_i|
      (0...y_component_number).each { |y_i|
        pr rescnn[v_i][y_i] === - tempcnn[0][v_i][y_i]     - tempcnn[1][v_i][y_i]                         - tempcnn[2][v_i][y_i]\
                                - tempcnn[0][v_i][y_i + 1] + tempcnn[1][v_i][y_i + 1] * "(#{temp_type})9" - tempcnn[2][v_i][y_i + 1]\
                                - tempcnn[0][v_i][y_i + 2] - tempcnn[1][v_i][y_i + 2]                     - tempcnn[2][v_i][y_i + 2]
        pr resnn[v_i][y_i] === clamp(rescnn[v_i][y_i],"(#{temp_type})0","(#{temp_type})255", :returns => rescnn[v_i][y_i])
      }
    }

    (0...(y_component_number)).each { |y_i|
      remaining_elem = total_x_size
      (0...vector_number).each { |v_i|
        if remaining_elem >= vector_length then
          pr pdst[x + v_i * vector_length, y + y_i] === resnn[v_i][y_i]
          remaining_elem -= vector_length
        else
          temp_vec_length = vector_length
          begin
            temp_vec_length = temp_vec_length/2
            elem_indexes = 0
            if remaining_elem >= temp_vec_length then
              pr pdst[x + (v_i * vector_length + elem_indexes), y + y_i] === resnn[v_i][y_i].components(elem_indexes...(elem_indexes+temp_vec_length))
              elem_indexes += temp_vec_length
              remaining_elem -= temp_vec_length
            end
          end while remaining_elem > 0
        end
      }
    }
  }
  pr p
  k.procedure = p
  return k
end

sizes = [[768, 432], [2560, 1600], [2048, 2048], [5760, 3240], [7680, 4320]]
inputs = []
refs = []
results = []
width = 1024
height = 512

set_lang(C)

k = laplacian_c_ref
puts k
sizes.each { |width, height|
  input = NArray.byte(width*3,height+1).random(256)
  output_ref = NArray.byte(width*3,height)

  k.run(width, height, input, output_ref)
  inputs.push(input)
  refs.push(output_ref[3..-4,1..-2])
  results.push( [] )
}

set_lang(CL)

opt_space = OptimizationSpace::new( :x_component_number => [1,2,4,8,16],
                                    :vector_length      => [1,2,4,8,16],
                                    :y_component_number => 1..4,
                                    :temporary_size     => [2,4],
                                    :vector_recompute   => [true, false],
                                    :load_overlap       => [true, false] )
check = false

optimizer = BruteForceOptimizer::new(opt_space, :randomize => true)

devs = OpenCL::platforms.select{ |p|
p.name.match(/Intel/)}.first.devices.first.partition_by_names_intel(
[0])
#[0,2,4,6,8,10,12,14,16,18,20,22] )
c = OpenCL::create_context( devs )


puts optimizer.optimize { |opt|
  id = opt.to_s            
  puts id
  k = laplacian(opt)
  puts k
  k.build( :CLCONTEXT => c )
  results = []
  sizes.each_index { |indx|
    GC.start
    width, height = sizes[indx]
    puts "#{width} x #{height} :"
    input_buff = k.context.create_buffer(width*height*3, :host_ptr => inputs[indx], :flags => [OpenCL::Mem::READ_ONLY,OpenCL::Mem::COPY_HOST_PTR])
    output_buff = k.context.create_buffer(width*height*3, :flags => [OpenCL::Mem::WRITE_ONLY])
    time_per_pixel=[]
    (0..3).each {
      stats = k.run(input_buff, output_buff, width, height, :global_work_size => [rndup((width*3/opt[:x_component_number].to_f).ceil,32), (height/opt[:y_component_number].to_f).ceil, 1], :local_work_size => [32, 1, 1])
      time_per_pixel.push( stats[:duration]/((width-2) * (height-2) ) )
    }
    #Fix for ARM counter looping every few minutes
    time_per_pixel.reject!{ |d| d < 0 }
    puts "#{time_per_pixel.min} s"

    if check then
      output = NArray.byte(width*3,height).random!(256)
      k.queue.enqueue_read_buffer( output_buff, output, :blocking => true)
      diff = ( refs[indx] - output[3..-4,1..-2] ).abs
      i = 0
      diff.each { |elem|
        #puts elem
        i += 1
        raise "Warning: residue too big: #{elem} #{i%3},#{(i / 3 ) % (width-2)},#{i / 3 / (width - 2)}" if elem > 0
      }
    end
    results.push( time_per_pixel.min )
  }
  results.reduce(:+) / results.length
}

exit

opt_space.each_random { |opt|
  id = opt.to_s            
  puts id
  k = laplacian(opt)
  puts k
  sizes.each_index { |indx|
    width, height = sizes[indx]
    puts "#{width} x #{height} :"
    output = NArray.byte(width*3,height).random!(256)
    durations=[]
    (0..3).each {
      stats = k.run(inputs[indx], output, width, height, :global_work_size => [rndup((width*3/opt[:x_component_number].to_f).ceil,32), (height/opt[:y_component_number].to_f).ceil, 1], :local_work_size => [32, 1, 1])
      durations.push stats[:duration]
    }
    #Fix for ARM counter looping every few minutes
    durations.reject!{ |d| d < 0 }
    puts "#{durations.min} s"

    if check then 
      diff = ( refs[indx] - output[3..-4,1..-2] ).abs
      i = 0
      diff.each { |elem|
        #puts elem
        i += 1
        raise "Warning: residue too big: #{elem} #{i%3},#{(i / 3 ) % (width-2)},#{i / 3 / (width - 2)}" if elem > 0
      }
    end
    results[indx].push( [id, durations.min] )
  }
}
puts "Best candidates:"
results.each_index { |indx|
  width, height = sizes[indx]
  puts "#{width} x #{height}"
  results[indx].sort! { |x,y| x[1] <=> y[1] }
  puts results[indx][0]
}
