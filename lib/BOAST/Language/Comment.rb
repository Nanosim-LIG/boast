module BOAST

  # @!parse module Functors; functorize Comment; end
  class Comment
    extend Functor
    include PrivateStateAccessor
    include Inspectable
    attr_reader :comment

    def initialize(string)
      @comment = string
    end

    def to_s
      return to_s_fortran if get_lang == FORTRAN
      return to_s_c if [C,CL,CUDA,HIP].include?(get_lang)
    end

    def pr
      s = to_s
      output.puts s
      return self
    end

    private

    def to_s_fortran
      s = ""
      @comment.each_line { |l| s << "! #{l}" }
      return s
    end

    def to_s_c
      s = ""
      @comment.each_line { |l| s << "/* #{l.delete("\n")} */\n" }
      return s
    end

  end

end
