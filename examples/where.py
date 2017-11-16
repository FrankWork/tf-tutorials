import tensorflow as tf
import numpy as np


# labels: 0:(e1,e2), 1:(e2,e1), 2:(other)
# self.rid:       5, 5, 7, 7, 1, 1
# self.direction: 0, 1, 0, 1, 0, 0
# labels i==5     0, 1, 2, 2, 2, 2      3 class
# labels i==7     2, 2, 0, 1, 2, 2      3 class
# labels i==1     1, 1, 1, 1, 0, 0      2 class

r_labels = [5,5,7,7,1,1]
d_labels = [0,1,0,1,0,0]

r_labels = tf.convert_to_tensor(r_labels, dtype=tf.float32)
d_labels = tf.convert_to_tensor(d_labels, dtype=tf.float32)

buf = []
for r in [5,7,1]:
  boolean = tf.equal(r_labels, r)
  x = d_labels
  y = tf.ones(x.shape.as_list())*2
  t_labels = tf.where(boolean, x, y)
  buf.append(t_labels)

with tf.Session() as sess:
  r,d = sess.run([r_labels, d_labels])
  print(r)
  print(d)
  print('-'*20)
  for b in sess.run(buf):
    print(b)