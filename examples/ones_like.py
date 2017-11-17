import tensorflow as tf
import numpy as np

a = tf.reshape(tf.range(10), [5, 2])
a1 = tf.ones(a.shape.as_list(), dtype=tf.int32)

b = tf.placeholder(tf.int32, [None, 2])
b1 = tf.ones_like(b)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  fetch = [a, a1, b, b1]
  for t in fetch:
    print(t.shape)

  for val in sess.run(fetch, {b: np.zeros((3,2))}):
    print(val)
    print('-'*80)