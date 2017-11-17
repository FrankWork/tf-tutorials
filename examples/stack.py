import tensorflow as tf


def axis():
  arr0 = tf.range(10)
  arr1 = tf.range(1, 11)
  idx0 = tf.stack([arr0, arr1], axis=0)
  idx1 = tf.stack([arr0, arr1], axis=1)
  print(arr0.shape)
  print(arr1.shape)
  print(idx0.shape)
  print(idx1.shape)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for val in sess.run([arr0, arr1, idx0, idx1]):
      print(val)
      print('-'*30)



axis()