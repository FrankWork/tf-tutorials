import tensorflow as tf
from example import parse_example
from sequence_example import parse_sequence_example

'''
tf.shuffle_batch(
    tensors,
    batch_size,
    capacity,
    min_after_dequeue,
    num_threads=1,
    seed=None,
    enqueue_many=False,
    shapes=None,
    allow_smaller_final_batch=False,
    shared_name=None,
    name=None
)
'''

if __name__ == '__main__':
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    name, age, scores = parse_example()
    # name, age, scores, bills = parse_sequence_example()
    name, age, scores = tf.train.shuffle_batch([name, age, scores], 3, 10, 3)

    init_op = tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer())

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      sess.run(init_op)
      print('-'*20 + ' example ' + '-'*20)

      try:
        while not sv.should_stop():
          n, a, s = sess.run([name, age, scores])
          print('-'*20)
          print(n)
          print(a)
          print(s)
      except tf.errors.OutOfRangeError:
        print('Done training')

  # with tf.Graph().as_default():
  #   name, age, scores, bills = parse_sequence_example()
  #   name, age, scores, bills = tf.train.shuffle_batch([name, age, scores, bills], 3, 10, 3)

  #   init_op = tf.group(tf.global_variables_initializer(),
  #                         tf.local_variables_initializer())

  #   sv = tf.train.Supervisor()
  #   with sv.managed_session() as sess:
  #     sess.run(init_op)
  #     print('-'*20 + ' sequence example ' + '-'*20)

  #     try:
  #       while not sv.should_stop():
  #         n, a, s, b = sess.run([name, age, scores, bills])
  #         print('-'*20)
  #         print(n)
  #         print(a)
  #         print(s)
  #         print(b)
  #     except tf.errors.OutOfRangeError:
  #       print('Done training')