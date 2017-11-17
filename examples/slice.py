import tensorflow as tf

arr0 = tf.reshape(tf.range(15), [5, 3])   # (5, 3)
s = arr0[:, :-1]
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  for val in sess.run([arr0, s]):
    print(val)
    print('-'*30)