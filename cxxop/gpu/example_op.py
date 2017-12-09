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
