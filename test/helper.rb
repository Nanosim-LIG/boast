module MiniTest::Assertions

  def assert_subprocess_output( stdout = nil, stderr = nil)
    out, err = capture_subprocess_io do
      yield
    end

    err_msg = Regexp === stderr ? :assert_match : :assert_equal if stderr
    out_msg = Regexp === stdout ? :assert_match : :assert_equal if stdout

    y = send err_msg, stderr, err, "In stderr" if err_msg
    x = send out_msg, stdout, out, "In stdout" if out_msg

    (!stdout || x) && (!stderr || y)
  end

end
