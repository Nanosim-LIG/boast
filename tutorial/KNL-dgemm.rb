
require 'BOAST'
include BOAST
set_lang(C)
set_array_start(0)
type_map = { 4 => NArray::SFLOAT, 8 => NArray::FLOAT}
register_funccall(:_mm_prefetch)

def micro_kernel(vector_length: 4, nr: nil, mr: nil, kb: nil, prefetch: nil)
  raise "nr must be a multiple of vector_length!" unless nr % vector_length == 0
  nvec = nr / vector_length
  register_number = mr * nvec + nvec
  puts "Using #{register_number} registers..."
  
  ah = Real :ah, dim: [Dim(mr), Dim(kb)], dir: :in, restrict: true
  bh = Real :bh, vector_length: vector_length, dim: [Dim(nvec), Dim(kb)], dir: :in, restrict: true
  ch = Real :ch, vector_length: vector_length, dim: [Dim(nvec), Dim(mr)], dir: :inout, restrict: true
  inp = Procedure("inp_#{vector_length}_#{nr}_#{mr}_#{kb}", [ah, bh, ch]) {
    regs = (0...nvec).collect { |n|
      (0...mr).collect { |m|
        Real :"reg_#{n}_#{m}", vector_length: vector_length, register: true
      }
    }
    regs_b = (0...nvec).collect { |n|
      Real :"regs_b_#{n}", vector_length: vector_length, register: true
    }
    decl *regs.flatten
    decl *regs_b
    (0...mr).collect { |m|
      (0...nvec).collect { |n|
        pr regs[n][m] === ch[n, m]
      }
    }
    i = Int :i
    decl i
    pr Pragma(:noprefetch) if prefetch
    pr Pragma(:unroll)
    pr For(i, 0, kb - 1) {
      (0...nvec).each { |n|
        pr _mm_prefetch(bh[n, i+8].address, 1) if prefetch
        pr regs_b[n] === bh[n, i]
      }
      (0...mr).each { |m|
        pr _mm_prefetch(ah[m, i+16].address, 1) if prefetch && m % 8 == 0
        (0...nvec).each { |n|
          pr regs[n][m] === FMA(Set(ah[m, i], regs_b[0]), regs_b[n], regs[n][m])
        }
      }
    }
    (0...mr).collect { |m|
      (0...nvec).collect { |n|
        pr ch[n, m] === regs[n][m]
      }
    }
  }
  inp.ckernel(includes: "immintrin.h")
end

vector_length = 8
mr = 31
nr = 8
kb = 464
nvec = nr / vector_length
type = type_map[get_default_real_size]
alignment = get_default_real_size*vector_length
a = NMatrix::new(type, kb, mr).random!
b = NMatrix::new(type, nr, kb).random!
c = NMatrix::new(type, nr, mr).random!

ah = ANArray::new(type, alignment, mr, kb)
bh = ANArray::new(type, alignment, vector_length, nvec, kb)
ch = ANArray::new(type, alignment, vector_length, nvec, mr)
c_ref = ANArray::new(type, alignment, vector_length, nvec, mr)
ah[true, true] = a.transpose(1,0)[true, true]
bh[true, true, true] = b.reshape(vector_length, nvec, kb)[true, true, true]
ch[true, true, true] = c.reshape(vector_length, nvec, mr)[true, true, true]
c_ref[true, true, true] = (a*b + c).reshape(vector_length, nvec, mr)[true, true, true]

p = micro_kernel(vector_length: vector_length, mr: mr, nr: nr, kb: kb, prefetch: true)
p.run(ah, bh, ch)
max_error = (ch - c_ref).abs.max
raise "Computation error!" if max_error > 1e-8
puts "Success!"
nil

p.run(ah, bh, ch)
repeat_inner = 100
res = 1000.times.collect {
  p.run(ah, bh, ch, repeat: repeat_inner)
}
best = res.min { |r1, r2|
  r1[:duration] <=> r2[:duration]
}
perf = mr * nr * kb * 2 / (best[:duration] * 1e9 / repeat_inner )
puts "time: #{best[:duration] / repeat_inner} s, GFlops: #{perf}"

def medium_kernel(vector_length: 4, mb: nil, nr: nil, mr: nil, kb: nil, prefetch: nil, openmp: nil)
  raise "nr must be a multiple of vector_length!" unless nr % vector_length == 0
  raise "mr must be a multiple of mb!" unless mb % mr == 0
  nvec = nr / vector_length
  nblocka = mb / mr

  inp = micro_kernel(vector_length: vector_length, nr: nr, mr: mr, kb: kb, prefetch: prefetch)
  
  #n = Int :n, dir: :in
  nblockn = Int :nblockn, dir: :in #n / nr
  at = Real :at, dim: [Dim(mr), Dim(kb), Dim(nblocka)], dir: :in, restrict: true
  bt = Real :bt, vector_length: vector_length, dim: [Dim(nvec), Dim(kb),  Dim(nblockn)], dir: :in, restrict: true
  ct = Real :ct, vector_length: vector_length, dim: [Dim(nvec), Dim(mr),  Dim(nblocka), Dim(nblockn)], dir: :inout, restrict: true
  medp = Procedure( "medp_#{vector_length}_#{mb}_#{nr}_#{mr}_#{kb}", [nblockn, at, bt, ct] ) {
    jr = Int :jr
    ir = Int :ir
    decl jr, ir
    pr OpenMP::ParallelFor( default: :shared, private: [jr, ir], num_threads: 4 ) if openmp
    pr For(jr, 0, nblockn - 1) {
      pr For(ir, 0, nblocka - 1) {
        pr inp.procedure.call(at[0, 0, ir].address, bt[0, 0, jr].address, ct[0, 0, ir, jr].address)
      }
    }
  }
  k = CKernel::new(includes: "immintrin.h") {
    pr inp.procedure
    pr medp
  }
  k.procedure = medp
  k
end

vector_length = 8
mr = 31
nr = 8
nblocka = 4
mb = mr * nblocka
kb = 464
nblockn = 257
n  = nr * nblockn
nvec = nr / vector_length

type = type_map[get_default_real_size]
alignment = get_default_real_size*vector_length
a = NMatrix::new(type, kb, mb).random!
b = NMatrix::new(type, n, kb).random!
c = NMatrix::new(type, n, mb).random!

at = ANArray::new(type, alignment, mr, kb, nblocka)
bt = ANArray::new(type, alignment, vector_length, nvec, kb, nblockn)
ct = ANArray::new(type, alignment, vector_length, nvec, mr, nblocka, nblockn)
c_ref = ANArray::new(type, alignment, vector_length, nvec, mr, nblocka, nblockn)
at[true, true, true] = a.reshape(kb, mr, nblocka).transpose(1, 0, 2)[true, true, true]
bt[true, true, true, true] = b.reshape(vector_length, nvec, nblockn, kb)
                              .transpose(0, 1, 3, 2)[true, true, true, true]
ct[true, true, true, true, true] = c.reshape(vector_length, nvec, nblockn, mr, nblocka)
                              .transpose(0, 1, 3, 4, 2)[true, true, true, true, true]
c_ref[true, true, true, true, true] = (a*b + c).reshape(vector_length, nvec, nblockn, mr, nblocka)
                                         .transpose(0, 1, 3, 4, 2)[true, true, true, true, true]

p = medium_kernel(vector_length: vector_length, mb: mb, mr: mr, nr: nr, kb: kb, prefetch: true)
p.run(nblockn, at, bt, ct)
max_error = (ct - c_ref).abs.max
raise "Computation error!" if max_error > 1e-8
puts "Success!"
nil

p.run(nblockn, at, bt, ct)
repeat_inner = 10
res = 10.times.collect {
  p.run(nblockn, at, bt, ct, repeat: repeat_inner)
}
best = res.min { |r1, r2|
  r1[:duration] <=> r2[:duration]
}
perf = mb * n * kb * 2 / (best[:duration] * 1e9 / repeat_inner )
puts "time: #{best[:duration] / repeat_inner} s, GFlops: #{perf}"

def large_kernel(vector_length: 4, mb: nil, nr: nil, mr: nil, kb: nil, prefetch: nil, openmp: nil)
  raise "nr must be a multiple of vector_length!" unless nr % vector_length == 0
  raise "mr must be a multiple of mb!" unless mb % mr == 0
  
  inp = micro_kernel(vector_length: vector_length, nr: nr, mr: mr, kb: kb, prefetch: prefetch)
  medp = medium_kernel(vector_length: vector_length, mb: mb, nr: nr, mr: mr, kb: kb, prefetch: prefetch, openmp: openmp)
  
  #m = Int :m, dir: :in
  #n = Int :n, dir: :in
  #k = Int :k, dir: :in
  nvec = nr / vector_length
  nblockm = Int :nblockm, dir: :in #m / mb
  nblocka = mb / mr
  nblockn = Int :nblockn, dir: :in #n / nr
  nblockk = Int :nblockk, dir: :in #k / kb
  a = Real :a, dim: [Dim(mr), Dim(kb), Dim(nblocka), Dim(nblockm), Dim(nblockk)], dir: :in, restrict: true
  b = Real :b, vector_length: vector_length, dim: [Dim(nvec), Dim(kb), Dim(nblockn), Dim(nblockk)], dir: :in, restrict: true
  c = Real :c, vector_length: vector_length, dim: [Dim(nvec), Dim(mr), Dim(nblocka), Dim(nblockn), Dim(nblockm)], dir: :inout, restrict: true
  larp = Procedure( "larp_#{vector_length}_#{mb}_#{nr}_#{mr}_#{kb}", [nblockm, nblockn, nblockk, a, b, c] ) {
    p = Int :p
    i = Int :i
    decl p, i
#    pr OpenMP::Parallel(default: :shared) {
#    pr OpenMP::Single() {
    pr For(p, 0, nblockk - 1) {
      pr OpenMP::ParallelFor( default: :shared, private: [i], num_threads: 16, schedule: [:dynamic,1] ) if openmp
      pr For(i, 0, nblockm - 1) {
#        pr OpenMP::Task(depend: {in: [a[:all, :all, :all, i, p],
#                                      b[:all, :all, :all, p]],
#                                 inout: [c[:all, :all, :all, :all, i]] } ) {
          pr medp.procedure.call(nblockn, a[0, 0, 0, i, p].address, b[0, 0, 0, p].address, c[0, 0, 0, 0, i].address)
#        }
      }
#    }
#    }
    }
  }
  
  k = CKernel::new(includes: "immintrin.h") {
    pr inp.procedure
    pr medp.procedure
    pr larp
  }
  k.procedure = larp
  k
end

vector_length = 8
mr = 31
nr = 8
nblocka = 4
mb = mr * nblocka
kb = 464
nblockn = 257
n  = nr * nblockn
nvec = nr / vector_length
nblockm = 16
m = mb * nblockm
nblockk = 4
k = kb * nblockk

type = type_map[get_default_real_size]
alignment = get_default_real_size*vector_length
a = NMatrix::new(type, k, m).random!
b = NMatrix::new(type, n, k).random!
c = NMatrix::new(type, n, m).random!

puts "a: #{m}x#{k} (#{m*k*get_default_real_size/(1e9)} GiB)"
puts "b: #{k}x#{n} (#{k*n*get_default_real_size/(1e9)} GiB)"
puts "c: #{m}x#{n} (#{m*n*get_default_real_size/(1e9)} GiB)"

ap = ANArray::new(type, alignment, mr, kb, nblocka, nblockm, nblockk)
bp = ANArray::new(type, alignment, vector_length, nvec, kb, nblockn, nblockk)
cp = ANArray::new(type, alignment, vector_length, nvec, mr, nblocka, nblockn, nblockm)
c_ref = ANArray::new(type, alignment, vector_length, nvec, mr, nblocka, nblockn, nblockm)
ap[true, true, true, true, true] = a.reshape(kb, nblockk, mr, nblocka, nblockm)
                                    .transpose(2, 0, 3, 4, 1)[true, true, true, true, true]
bp[true, true, true, true, true] = b.reshape(vector_length, nvec, nblockn, kb, nblockk)
                                    .transpose(0, 1, 3, 2, 4)[true, true, true, true, true]
cp[true, true, true, true, true, true] = c.reshape(vector_length, nvec, nblockn, mr, nblocka, nblockm)
                                          .transpose(0, 1, 3, 4, 2)[true, true, true, true, true, true]
c_ref[true, true, true, true, true, true] = (a*b + c).reshape(vector_length, nvec, nblockn, mr, nblocka, nblockm)
                                                     .transpose(0, 1, 3, 4, 2)[true, true, true, true, true, true]
push_env(use_vla: true) {
  puts p = large_kernel(vector_length: vector_length, mb: mb, mr: mr, nr: nr, kb: kb, prefetch: true, openmp: true)
  p.build(openmp: true)
}
p.run(nblockm, nblockn, nblockk, ap, bp, cp)
  
max_error = (cp - c_ref).abs.max
raise "Computation error!" if max_error > 1e-8
puts "Success!"
nil

p.run(nblockm, nblockn, nblockk, ap, bp, cp)
res = 100.times.collect {
  p.run(nblockm, nblockn, nblockk, ap, bp, cp)
}
best = res.min { |r1, r2|
  r1[:duration] <=> r2[:duration]
}
perf = m * n * k * 2 / (best[:duration] * 1e9)
puts "time: #{best[:duration]} s, GFlops: #{perf}"
