import tensorflow as tf

a = tf.reshape(tf.range(10), [2,5])
p = tf.pow(a,2)
s = tf.square(a)

with tf.Session() as sess:
  for v in sess.run([a, s, p]):
    print(v)
    print('-'* 40)
