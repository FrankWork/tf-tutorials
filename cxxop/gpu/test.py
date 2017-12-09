import os
import tensorflow as tf

real_path = os.path.dirname(os.path.realpath(__file__))
op_path = os.path.join(real_path, 'example_op.so')


class MyTest(tf.test.TestCase):
  def testGRLOp(self):
    example_module = tf.load_op_library(op_path)

    x = tf.random_normal([3, 4])
    y = example_module.example_op(x)

    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      x, y = sess.run([x, y])

      self.assertAllEqual(x, y)
      
if __name__ == "__main__":
  tf.test.main()
