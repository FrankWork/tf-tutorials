import tensorflow as tf

'''
message BytesList {
  repeated bytes value = 1;
}
message FloatList {
  repeated float value = 1 [packed = true];
}
message Int64List {
  repeated int64 value = 1 [packed = true];
}

message Feature {
  // Each feature can be exactly one kind.
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
  }
};

message Features {
  // Map from feature name to feature.
  map<string, Feature> feature = 1;
};

message FeatureList {
  repeated Feature feature = 1;
};

'''

def repeated():
  '''
  message BytesList {
    repeated bytes value = 1;
  }
  message FloatList {
    repeated float value = 1 [packed = true];
  }
  message Int64List {
    repeated int64 value = 1 [packed = true];
  }
  '''
  arr = tf.train.Int64List()
  arr.value.append(15)         # Appends one value
  arr.value.extend([32, 47])   # Appends an entire list

  assert len(arr.value) == 3
  assert arr.value[0] == 15
  assert arr.value[1] == 32
  assert arr.value == [15, 32, 47]

  arr.value[:] = [33, 48]      # Assigns an entire list
  assert arr.value == [33, 48]

  arr.value[1] = 56            # Reassigns a value
  assert arr.value[1] == 56
  for i in arr.value:          # Loops and print
    print(i)
  del arr.value[:]             # Clears list (works just like in a Python list)

def oneof():
  '''
  message Feature {
    // Each feature can be exactly one kind.
    oneof kind {
      BytesList bytes_list = 1;
      FloatList float_list = 2;
      Int64List int64_list = 3;
    }
  };
  '''
  f = tf.train.Feature()
  f.int64_list.value.append(15)
  assert f.HasField('int64_list')
  assert not f.HasField('float_list')
  print(f.int64_list.value[0])

  f.float_list.value.append(15.0)
  assert not f.HasField('int64_list')
  assert f.HasField('float_list')
  print(f.float_list.value[0])

def map():
  '''
  message Features {
    // Map from feature name to feature.
    map<string, Feature> feature = 1;
  };
  '''
  f = tf.train.Features()
  f.feature['feature1'].int64_list.value.append(15)
  f.feature['feature2'].float_list.value.extend([16.0, 17.0])
  
  for k in f.feature:
    print(k)
    print(f.feature[k])
  
  del f.feature['feature1']

def repeated_fields():
  '''
  message FeatureList {
    repeated Feature feature = 1;
  };
  '''
  arr = tf.train.FeatureList()
  f = arr.feature.add()                         # Adds a Feature then modify
  f.int64_list.value.append(15)

  arr.feature.add().int64_list.value.append(32) # Adds and modify at the same time

  f = tf.train.Feature()
  f.int64_list.value.append(47)
  arr.feature.extend([f])                       # Uses extend() to copy
  print(arr.feature[0].int64_list.value[0])
  print(arr.feature[1].int64_list.value[0])
  print(arr.feature[2].int64_list.value[0])


print('-- repeated --')
repeated()
print('-- oneof --')
oneof()
print('-- map --')
map()
print('-- repeated fields --')
repeated_fields()
