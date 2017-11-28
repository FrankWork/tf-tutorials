import tensorflow as tf
import numpy as np

a = tf.truncated_normal([3,4])
b = tf.truncated_normal([4])

c = a + b
d = tf.nn.bias_add(a, b)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  c, d = sess.run([c,d])
  e = np.array_equal(c,d)
  print(e)
