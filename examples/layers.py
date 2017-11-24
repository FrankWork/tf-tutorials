import tensorflow as tf

K = tf.keras

class KerasLinearLayer(K.layers.Layer):

  def __init__(self, out_size, **kwargs):
    self.out_size = out_size
    super(KerasLinearLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    with tf.device('/cpu:0'):
      self.w = self.add_weight('W', [input_shape[1], self.out_size])
      self.b = self.add_weight('b', [self.out_size])
    super(KerasLinearLayer, self).build(input_shape)
  
  def call(self, x):
    y = tf.nn.xw_plus_b(x, self.w, self.b)
    return y

class TensorflowLinearLayer(tf.layers.Layer):

  def __init__(self, out_size, **kwargs):
    self.out_size = out_size
    super(TensorflowLinearLayer, self).__init__(**kwargs)
  
  def build(self, input_shape):
    with tf.device('/cpu:0'):
      self.w = self.add_variable('W', [input_shape[1], self.out_size])
      self.b = self.add_variable('b', [self.out_size])
    super(TensorflowLinearLayer, self).build(input_shape)
  
  def call(self, x):
    y = tf.nn.xw_plus_b(x, self.w, self.b)
    return y

tf.set_random_seed(1)
x = tf.random_normal([3,4])

keras_linear = KerasLinearLayer(5)
y_keras = keras_linear(x)

tf_linear = TensorflowLinearLayer(5)
y_tf = tf_linear(x)

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for v in sess.run([y_keras, y_tf]):
    print(v.shape)
    print(v)
    print('-' * 40)