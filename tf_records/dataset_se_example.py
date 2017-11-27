import tensorflow as tf
import random

se_example_file = "se_example.tfrecords"

def read_records(filename):
  reader = tf.python_io.tf_record_iterator(filename)
  records = []
  for record in reader:
    # record is of <class 'bytes'>
    records.append(record)
  return records

def write_records(records, out_filename):
  writer = tf.python_io.TFRecordWriter(out_filename)
  for count, record in enumerate(records):
    writer.write(record)
  writer.close()

def shuf_and_write(records, filename):
  records = read_records(filename)
  random.shuffle(records)
  write_records(records, filename+".shuf")

def parse_sequence_example(serialized_example):
  context_features={
                      'name'   : tf.FixedLenFeature([], tf.string),
                      'age'    : tf.FixedLenFeature([], tf.int64)}
  sequence_features={
                      'scores' : tf.FixedLenSequenceFeature([], tf.int64),
                      'bills'  : tf.FixedLenSequenceFeature([], tf.float32)}
  context_dict, sequence_dict = tf.parse_single_sequence_example(
                      serialized_example,
                      context_features   = context_features,
                      sequence_features  = sequence_features)

  # tensors
  name = context_dict['name']
  # name = tf.decode_raw(name, tf.uint8)
  age = context_dict['age']

  scores = sequence_dict['scores']
  bills = sequence_dict['bills']

  return name, age, scores, bills # #

parse_func = parse_sequence_example
filename = se_example_file

with tf.Graph().as_default():
  dataset = tf.data.TFRecordDataset([filename])
  dataset = dataset.map(parse_func)  # Parse the record into tensors.
  # dataset = dataset.shuffle(buffer_size=100)
  dataset = dataset.repeat(3)        # Number of epoches
  dataset = dataset.padded_batch(3, padded_shapes=([], [], [None], [None]))
  iterator = dataset.make_one_shot_iterator()
  # a list
  next = iterator.get_next()

  print(dataset.output_shapes)
  print(dataset.output_types)
  with tf.train.MonitoredTrainingSession() as sess:
    print('-'*20 + ' sequence example ' + '-'*20)
    while not sess.should_stop():
      print('-'*20)
      _, _, _, bills = sess.run(next)
      print(bills.shape)