require 'BOAST'
module BOAST
  def BOAST::SimpleLoop(unroll)
    i = Variable::new('i', Int)
    sum = Variable::new('sum', Int)
    data = Variable::new("data",Int, :dimension => [ Dimension::new(10)])
    pr sum === 0
    pr For::new(i, 0, 9, step: unroll) {
        1.upto(unroll) { |index|
          pr sum === sum + data[i + index]
        }
    }
  end
end

