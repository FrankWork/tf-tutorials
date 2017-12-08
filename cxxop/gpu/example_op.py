# compile:
# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
# g++ -std=c++11 -shared grl_op.cc grl_kernel.cc -o grl_op.so -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2

import os
import tensorflow as tf

op_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'example_op.so')
example_op_module = tf.load_op_library(op_path)


class ExampleOpTest(tf.test.TestCase):
  def testExampleOp(self):
    with self.test_session():
      x = tf.random_normal([3, 4])
      y = 2*x
      result = example_op_module.example_op(x)
      y, result = sess.run([y, result])
      self.assertAllEqual(y, result)

if __name__ == "__main__":
  tf.test.main()
