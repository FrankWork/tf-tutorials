import tensorflow as tf

example_file = "example.tfrecords"


def parse_example(serialized_example):
  example_features={
                      'name'  : tf.FixedLenFeature([], tf.string),
                      'age'   : tf.FixedLenFeature([], tf.int64),
                      'scores': tf.FixedLenFeature([3], tf.float32),}
  feature_dict = tf.parse_single_example(
                      serialized_example,
                      example_features)

  name = feature_dict['name']
  # name = tf.decode_raw(name, tf.uint8)
  age = feature_dict['age']
  scores = feature_dict['scores']

  return name, age, scores



parse_func = parse_example
filename = example_file

with tf.Graph().as_default():
  dataset = tf.data.TFRecordDataset([filename])
  dataset = dataset.map(parse_func)  # Parse the record into tensors.
  # dataset = dataset.shuffle(buffer_size=100)

  dataset = dataset.repeat(1)        # Number of epoches
  dataset = dataset.batch(3)
  iterator = dataset.make_one_shot_iterator()
  next = iterator.get_next()

  with tf.train.MonitoredTrainingSession() as sess:
    print('-'*20 + ' example ' + '-'*20)
    while not sess.should_stop():
      name, age, scores = sess.run(next)
      print('-'*20)
      print(name)
      print(age)
      print(scores)

