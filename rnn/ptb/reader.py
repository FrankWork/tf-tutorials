# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf


def _read_words(filename):
  # tf.gfile.GFile(...)
  #     File I/O wrappers without thread locking
  #
  # read(self, n=-1)
  #     Returns the contents of a file as a string.
  #     Starts reading from current position in file.

  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  # Elements are stored as dictionary keys and their counts
  # are stored as dictionary values
  counter = collections.Counter(data)
  # counter.items() => dict_items([('a', 3), ('b', 2), ('c', 1)])
  # key => (-3, 'a')
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """
      Load PTB raw data from data directory "data_path".

      Reads PTB text files, converts strings to integer ids,
      and performs mini-batching of the inputs.

      The PTB dataset comes from Tomas Mikolov's webpage:

      http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

      Args:
        data_path: string path to the directory where simple-examples.tgz has
          been extracted.

      Returns:
        tuple (train_data, valid_data, test_data, vocabulary)
        where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls(展开).
               see the reader_test.py to get an intuition
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  # name_scope(name, default_name=None, values=None)
  #     Returns a context manager for use when defining a Python op.
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size # number of batches to generated
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    # tf.control_dependencies(control_inputs)
    #   Wrapper for `Graph.control_dependencies()` using the default graph.
    #
    #   Args:
    #       control_inputs: A list of `Operation` or `Tensor` objects which
    #       must be executed or computed before running the operations
    #       defined in the `with` context.  Can also be `None` to clear the control
    #       dependencies.
    with tf.control_dependencies([assertion]):
      # identity(input, name=None)
      #    Return a tensor with the same shape and contents as the input tensor or value.
      #
      # epoch_size: an int32 scalar tensor
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    #     tf.trian.range_input_producer(limit, num_epochs=None, shuffle=True)
    #       Produces the integers from 0 to limit-1 in a queue.
    #
    #     Note: if `num_epochs` is not `None`, this function creates local counter
    #     `epochs`. Use `local_variables_initializer()` to initialize local variables.
    #
    #     Args:
    #       limit: An int32 scalar tensor.
    #       num_epochs: An integer (optional). If specified, `range_input_producer`
    #         produces each integer `num_epochs` times before generating an
    #         OutOfRange error. If not specified, `range_input_producer` can cycle
    #         through the integers an unlimited number of times.
    #       shuffle: Boolean. If true, the integers are randomly shuffled within each
    #         epoch.
    #
    # dequeue()
    #   i is an int
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    # tf.strided_slice(input_, begin, end, strides=None)
    # Extracts a strided slice from a tensor.
    #
    # To a first order, this operation extracts a slice of ** size `end - begin` **
    # from a tensor `input`
    # starting at the location specified by `begin`. The slice continues by adding
    # `stride` to the `begin` index until all dimensions are not less than `end`.
    # Note that components of stride can be negative, which causes a reverse
    # slice.
    #
    # This operation can be thought of an encoding of a numpy style sliced
    # range. Given a python slice input[<spec0>, <spec1>, ..., <specn>]
    # this function will be called as follows.
    #
    # `begin`, `end`, and `strides` will be all length n. n is in general
    # not the same dimensionality as `input`.
    # NOTE: `begin` and `end` are zero-indexed`.
    # `strides` entries must be non-zero.
    #
    # ```python
    # # 'input' is [[[1, 1, 1], [2, 2, 2]],
    # #             [[3, 3, 3], [4, 4, 4]],
    # #             [[5, 5, 5], [6, 6, 6]]]
    # # shape: (3,2,3)
    # tf.strided_slice(input, [1, 0, 0], [2, 1, 3], [1, 1, 1]) ==> [[[3, 3, 3]]]
    # tf.strided_slice(input, [1, 0, 0], [2, 2, 3], [1, 1, 1]) ==> [[[3, 3, 3],
    #                                                                [4, 4, 4]]]
    # tf.strided_slice(input, [1, 1, 0], [2, -1, 3], [1, -1, 1]) ==>[[[4, 4, 4],
    #                                                                 [3, 3, 3]]]
    # ```
    #
    # Args:
    #   input_: A `Tensor`.
    #   begin: An `int32` or `int64` `Tensor`.
    #   end: An `int32` or `int64` `Tensor`.
    #   strides: An `int32` or `int64` `Tensor`.
    # Returns:
    #   A `Tensor` the same type as `input`.

    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y
