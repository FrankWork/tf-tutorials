import tensorflow as tf

a = tf.reshape(tf.range(10), [2,5])
s = tf.pow(a,2)

with tf.Session() as sess:
  for v in sess.run([a, s]):
    print(v)
    print('-'* 40)
