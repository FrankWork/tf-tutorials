import tensorflow as tf

'''
message Example {
  Features features = 1;
};

parse_single_example(
  serialized, 
  features, 
  name=None, 
  example_names=None
)

string_input_producer(
    string_tensor,
    num_epochs=None,
    shuffle=True,
    seed=None,
    capacity=32,
    shared_name=None,
    name=None,
    cancel_op=None
)
'''

N_example = 10
data_file = "example.tfrecords"

def build_example(name, age, scores):
  '''
  Args: 
    name  : string
    age   : int
    scores: a fix-sied list [1., 2., 3.], 
            reprents the scores for math, english, cs

  Returns:
    tf.trian.Example
  '''
  ex = tf.train.Example()

  name_bytes = bytes(name, 'utf8')
  ex.features.feature['name'].bytes_list.value.append(name_bytes)

  ex.features.feature['age'].int64_list.value.append(age)

  ex.features.feature['scores'].float_list.value.extend(scores)

  return ex

def parse_example():
  filename_queue = tf.train.string_input_producer(
                        [data_file], 
                        num_epochs=1)
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)
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

  return (name, age, scores)

if __name__ == '__main__':

  # write TFRecord to data_file
  writer = tf.python_io.TFRecordWriter(data_file)
  for i in range(N_example):
    name = 'a'*i
    age = 20 + i
    scores = [float(x*i) for x in range(1, 4)]
    example = build_example(name, age, scores)
    writer.write(example.SerializeToString())
  writer.close()

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    
    name, age, scores = parse_example()
    init_op = tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer())

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      sess.run(init_op)
      
      try:
        while not sv.should_stop():
          n, a, s = sess.run([name, age, scores])
          print('-'*20)
          print(n)
          print(a)
          print(s)
      except tf.errors.OutOfRangeError:
        print('Done training')
