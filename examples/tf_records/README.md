# TensorFlow TFRecord Format Tutorials

Tutorials on TFRecord, Example, SequenceExample

## feature.py 

- `tf.train.Int64List` : Protocol Buffers `repeated` scalar field
- `tf.train.FloatList` : Protocol Buffers `repeated` scalar field
- `tf.train.BytesList` : Protocol Buffers `repeated` scalar field
- `tf.train.Feature`   : Protocol Buffers `oneof` field
- `tf.train.Features`  : Protocol Buffers `map` field
- `tf.train.FeatureList`  : Protocol Buffers `repeated` message field

## example.py

- `tf.train.Example` :   A wrapper of `Features`

## sequence_example.py

- `tf.train.FeatureList`    : Protocol Buffers `repeated` field
- `tf.train.FeatureLists`   : Protocol Buffers `map` field
- `tf.train.SequenceExample`: A wrapper of `Features` and `FeatureLists`

## batch.py

Batch `Example` or `SequenceExample`

## shuffle_batch.py

Shuffle batch `Example`

## dataset.py

`tf.data.Dataset` to shuffle and batch `Example` and `SequenceExample`