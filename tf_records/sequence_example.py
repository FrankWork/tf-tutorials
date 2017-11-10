import tensorflow as tf

'''
message FeatureList {
  repeated Feature feature = 1;
};

message FeatureLists {
  // Map from feature name to feature list.
  map<string, FeatureList> feature_list = 1;
};

message SequenceExample {
  Features context = 1;
  FeatureLists feature_lists = 2;
};

tf.parse_single_sequence_example(
    serialized,
    context_features=None,
    sequence_features=None,
    example_name=None,
    name=None
)
'''

N_example = 10
data_file = "se_example.tfrecords"

def build_sequence_example(name, age, scores, bills):
  '''
  Args: 
    name  : string
    age   : int
    scores: a list of int   [1, 2, ...]
    bills : a list of float [1., 2., ...]

  Returns:
    tf.trian.SequenceExample
  '''
  ex = tf.train.SequenceExample()

  name_bytes = bytes(name, 'utf8')
  ex.context.feature['name'].bytes_list.value.append(name_bytes)
  ex.context.feature['age'].int64_list.value.append(age)

  for score in scores:
    # type(score_feature) is tf.train.Feature
    score_feature = ex.feature_lists.feature_list['scores'].feature.add()
    score_feature.int64_list.value.append(score)

  for bill in bills:
    bill_feature = ex.feature_lists.feature_list['bills'].feature.add()
    bill_feature.float_list.value.append(bill)
 
  return ex

def parse_sequence_example():
  filename_queue = tf.train.string_input_producer(
                      [data_file], 
                      num_epochs=1)
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)
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

  name = context_dict['name']
  # name = tf.decode_raw(name, tf.uint8)
  age = context_dict['age']

  scores = sequence_dict['scores']
  bills = sequence_dict['bills']

  return (name, age, scores, bills)


if __name__ == '__main__':
  # write TFRecord to data_file
  writer = tf.python_io.TFRecordWriter(data_file)
  for i in range(1, N_example+1):
    name = 'a'*i
    age = 20 + i
    scores = [x*i for x in range(i)]
    bills = [float(x*i*i) for x in range(i)]
    example = build_sequence_example(name, age, scores, bills)
    writer.write(example.SerializeToString())
  writer.close()

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    name, age, scores, bills = parse_sequence_example()
    init_op = tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer())

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      sess.run(init_op)
      
      try:
        while not sv.should_stop():
          n, a, s, b = sess.run([name, age, scores, bills])
          print('-'*20)
          print(n)
          print(a)
          print(s)
          print(b)
      except tf.errors.OutOfRangeError:
        print('Done training')
