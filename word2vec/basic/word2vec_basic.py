import tensorflow as tf
import numpy as np
import os
from six.moves import urllib, xrange
import zipfile
import collections
import random
import math

# def singleton(cls, *args, **kw):
#     """
#     singleton decorator
#     单例模式装饰器
#     """
#     instances = {}
#     def _singleton(cls, *args, **kw):
#         if cls not in instances:
#             instances[cls] = cls(*args, **kw)
#         return instances[cls]
#     return _singleton
#
# @singleton
class DataManager(object):
    _url = 'http://mattmahoney.net/dc/'
    _filename = 'text8.zip'
    _expected_bytes = 31344016
    _data_index = 0

    vocabulary_size = 50000
    batch_size = 8
    num_skips = 2   # # How many times to reuse an input to generate a label.
    skip_window = 1 # How many words to consider left and right.

    data = None
    count = None
    dictionary = None
    reverse_dictionary = None

    def __init__(self, batch_size=8,num_skips=2,skip_window=1):
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window

        self._check_and_get_file()
        words = self._read_data()
        self._build_dataset(words)
        del words # Hint to reduce memory

    def _check_and_get_file(self):
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.exists(self._filename) :
            self._filename, _ = urllib.request.urlretrieve(_url + self._filename, self._filename)
        statinfo = os.stat(self._filename)
        if statinfo.st_size == self._expected_bytes:
            print('Found and verified', self._filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                'Failed to verify ' + self._filename +
                '. Can you get to it with a browser?'
            )
        return self._filename

    def _read_data(self):
        """
        Extract the first file enclosed in a zip file as a list of words

        tf.compat.as_str(bytes_or_text)
            Returns the given argument as a unicode string
        class ZipFile(builtins.object)
            Class with methods to open, read, write, close, list zip filesself.
            read(name)
                Return file bytes (as a string) for name

        """
        with zipfile.ZipFile(self._filename) as f:
            self.data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return self.data

    def _build_dataset(self, words):
        """
        Build the dictionary and replace rare words with UNK token.

        extend(...) method of builtins.list instance
            extend list by appending elements from the iterable
        class Counter(builtins.dict)
          Dict subclass for counting hashable items.  Sometimes called a bag
          or multiset.  Elements are stored as dictionary keys and their counts
          are sorted and stored as dictionary values.

        """
        self.count = [['UNK', -1]]
        self.count.extend(collections.Counter(words).most_common(self.vocabulary_size - 1))
        self.dictionary = dict()
        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        self.data = list()
        unk_count = 0
        for word in words:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = 0
                unk_count += 1
            self.data.append(index)
        self.count[0][1] = unk_count
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        return self.data, self.count, self.dictionary, self.reverse_dictionary

    def generate_batch(self):
        """
            Function to generate a training batch for the skip-gram model.

            :type num_skips: int
            :param num_skips: how many (target, word-in-context) pairs be generated

            :type skip_window: int
            :param skip_window: left window size, right window size

            As an example, let's consider the dataset
              the quick brown fox jumped over the lazy dog
            Define 'context' as the window of words to the left and to the right of a
            target word. Using a window size of 1, we then have the dataset
              ([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox)
            of (context, target) pairs.

            Recall that skip-gram inverts contexts and targets, and tries to predict each
            context word from its target word, so the task becomes to predict 'the' and
            'brown' from 'quick', 'quick' and 'fox' from 'brown', etc. Therefore our
            dataset becomes
              (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
            of (input, output) pairs.
            https://www.tensorflow.org/tutorials/word2vec

            class deque(builtins.object)
                deque([iterable[, maxlen]]) --> deque object
                    A list-like sequence optimized for data accesses near its endpoints.

            random.randint(a, b) method of random.Random instance
                Return random integer in range [a, b], including both end points.

        """
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        span = 2 * self.skip_window + 1 # [skip_window target skip_window]
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32) # row vector
        labels = np.ndarray(shape=(self.batch_size,1), dtype=np.int32) # column vector
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self._data_index])
            self._data_index = (self._data_index + 1) % len(self.data)
        for i in range(self.batch_size // self.num_skips):
            target = self.skip_window # target now is the center of the buffer
            targets_to_avoid = [target]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1) # choose a word in context
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[target]
                targets_to_avoid.append(target)
            buffer.append(self.data[self._data_index])
            self._data_index = (self._data_index + 1) % len(self.data)
        return batch, labels

    def generate_validation_set(self):
        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_size = 16     # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        # np.random.choice(a, size=None, replace=True, p=None)
        #
        #     Generates a random sample from a given 1-D array
        #
        #     Parameters
        #     -----------
        #     a : 1-D array-like or int
        #         If an ndarray, a random sample is generated from its elements.
        #         If an int, the random sample is generated as if a was np.arange(n)
        #     size : int or tuple of ints, optional
        #         Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        #         ``m * n * k`` samples are drawn.  Default is None, in which case a
        #         single value is returned.
        #     replace : boolean, optional
        #         Whether the sample is with or without replacement
        #     Examples
        #     ---------
        #
        #     >>> np.random.choice(5, 3)
        #     array([0, 3, 4])
        return valid_examples

    def print_batch(self, batch, labels):
        for i in range(self.batch_size):
          print(batch[i], self.reverse_dictionary[batch[i]],
                '->', labels[i, 0], self.reverse_dictionary[labels[i, 0]])

class SkipGramModel(object):
    batch_size = 128
    embedding_size = 128 # Dimension of the embedding vector.
    skip_window = 1
    num_skips = 2
    num_sampled = 64 # Number of negative examples to sample.
    vocabulary_size = 50000
    num_steps = 100001

    def __init__(self):
        self.data_mgr = DataManager(self.batch_size, self.num_skips, self.skip_window)

    def build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            # Input Data
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.valid_examples = self.data_mgr.generate_validation_set()
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            with tf.device('/cpu:0'): # The CPU of your machine
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, self.embedding_size],
                            -1.0,1.0))
                # size: batch_size x embedding_size
                embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)


                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                        stddev=1.0 / math.sqrt(self.embedding_size)))
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

                # Compute the average NCE loss for the batch.
                # tf.nce_loss automatically draws a new sample of the negative labels each
                # time we evaluate the loss.
                self.loss = tf.reduce_mean(
                         tf.nn.nce_loss(weights=nce_weights,
                                        biases=nce_biases,
                                        labels=self.train_labels,
                                        inputs=embed,
                                        num_sampled=self.num_sampled,
                                        num_classes=self.vocabulary_size))
                #  tf.reduce_mean
                #     'x' is [[1., 1.]
                #             [2., 2.]]
                #     tf.reduce_mean(x) ==> 1.5
                #     tf.reduce_mean(x, 0) ==> [1.5, 1.5]
                #     tf.reduce_mean(x, 1) ==> [1.,  2.]
                #
                # tf.nn.nce_loss
                #     Computes and returns the noise-contrastive estimation training loss.

                # Construct the SGD optimizer using a learning rate of 1.0.
                self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

                # Compute the cosine similarity between minibatch examples and all embeddings.
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                self.normalized_embeddings = embeddings / norm
                valid_embeddings = tf.nn.embedding_lookup(
                    self.normalized_embeddings, valid_dataset)
                self.similarity = tf.matmul(
                    valid_embeddings, self.normalized_embeddings, transpose_b=True)
                    #  tf.matmul(a, b, transpose_a=False, transpose_b=False, ...)
        return graph

    def build_session(self, graph):
        with tf.Session(graph=graph) as session:
            # Add variable initializer.
            init = tf.global_variables_initializer()
            init.run()
            print('Initialized')

            average_loss = 0
            for step in range(self.num_steps):
                batch_inputs, batch_labels = self.data_mgr.generate_batch()
                feed_dict = {self.train_inputs:batch_inputs,self.train_labels:batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([self.optimizer, self.loss],feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = self.similarity.eval()
                    for i in range(self.data_mgr.valid_size):
                        valid_word = self.data_mgr.reverse_dictionary[self.valid_examples[i]]
                        top_k = 8 # number of nearest neighbors
                        nearest = (-sim[i,:]).argsort()[1:top_k + 1]
                        log_str = "nearest to %s:" % valid_word
                        for k in range(top_k):
                            close_word = self.data_mgr.reverse_dictionary[nearest[k]]
                            log_str = "%s, %s" % (log_str, close_word)
                        print(log_str)
            final_embeddings = self.normalized_embeddings.eval()
            return final_embeddings

    def train(self):
        graph = self.build_graph()
        return self.build_session(graph)
    def plot(self, final_embeddings):
        try:
          from sklearn.manifold import TSNE
          import matplotlib.pyplot as plt

          tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
          plot_only = 500
          low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
          labels = [reverse_dictionary[i] for i in xrange(plot_only)]
          _plot_with_labels(low_dim_embs, labels)

        except ImportError:
          print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")


    def _plot_with_labels(self, low_dim_embs, labels, filename='tsne.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(filename)



if __name__ == '__main__':
    model = SkipGramModel()
    final_embeddings = model.train()
    model.plot(final_embeddings)
