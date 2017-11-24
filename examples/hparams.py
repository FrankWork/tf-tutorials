import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('a_param', 'in a', 'Which dataset to generate data for')
flags.DEFINE_integer('rnn_size', None, 'rnn size')

hparams = tf.contrib.training.HParams(
                                learning_rate=0.1, 
                                num_hidden_units=100,
                                activations=['relu', 'tanh'])

print(hparams.learning_rate)

hparams.add_hparam('rnn_size', 110)

print(hparams.rnn_size)

def main(_):
  if FLAGS.a_param:
    print(FLAGS.a_param)
  
if __name__ == '__main__':
  tf.app.run()