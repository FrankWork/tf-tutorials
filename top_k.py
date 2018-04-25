import numpy as np
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

labels = tf.constant([0, 1, 2, 2, 1], dtype=tf.int64)
pred = tf.constant([
    [0.8, 0.1, 0.1],
    [0.4, 0.2, 0.4],
    [0.1, 0.3, 0.7],
    [0.4, 0.5, 0.1],
    [0.1, 0.6, 0.3]
], dtype=tf.float32)

mask = tf.constant([0, 1, 1], dtype=tf.float32)

# print(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), tf.float32)).numpy())
# print(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred*mask, axis=1), labels), tf.float32)).numpy())

k=5
_, idx = tf.nn.top_k(tf.reshape(pred*mask, [-1]), k)
label = tf.gather( tf.reshape(tf.one_hot(labels, 3), [-1]), idx)
p = tf.reduce_mean(label)
# print(p.numpy())




# metric = tf.metrics.precision_at_k(tf.expand_dims(labels, 0), 
#     tf.reshape(pred*mask, [1, -1]), k)
# with tf.Session() as sess:
#   sess.run(tf.local_variables_initializer())
#   print(sess.run(metric)[1])
    