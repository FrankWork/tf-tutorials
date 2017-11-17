import tensorflow as tf
import numpy as np


labels = tf.reshape(tf.range(0, 19), [19, 1])
task = tf.cast(labels / 2, tf.int32)
direc = labels % 2

t = tf.concat([labels, task, direc], 1)

with tf.Session() as sess:
  for v in sess.run([t]):
    print(v)
    print('-'*70)