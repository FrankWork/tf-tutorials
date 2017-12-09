import os
import tensorflow as tf
import grl_gradient

op_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'grl_op.so')


class MyTest(tf.test.TestCase):
  def testGRLOp(self):
    grl_module = tf.load_op_library(op_path)

    x = tf.random_normal([3, 4])
    w = tf.get_variable('w', [4,2], dtype=tf.float32)
    b = tf.get_variable('b', [2], dtype=tf.float32)

    y = tf.nn.xw_plus_b(x, w, b)
    g = tf.gradients(y, [w, b])

    y2 = tf.nn.xw_plus_b(x, grl_module.grl_op(w), b)
    g2 = tf.gradients(y2, [w, b])

    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      y, y2, g, g2 = sess.run([y, y2, g, g2])

      self.assertAllEqual(y, y2)
      self.assertAllEqual(g[0], -g2[0]) # gradient of w
      self.assertAllEqual(g[1], g2[1]) # gradient of b
      
if __name__ == "__main__":
  tf.test.main()

  