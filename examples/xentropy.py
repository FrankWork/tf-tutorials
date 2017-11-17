import tensorflow as tf


labels = tf.range(5)
labels = tf.one_hot(labels, 3)

logits = tf.reshape(tf.range(15, dtype=tf.float32), [5, 3])

xentropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels, 
                    logits=logits))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for v in sess.run([xentropy]):
    print(v)
    print('-'*80)
