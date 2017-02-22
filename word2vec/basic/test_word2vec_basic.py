import tensorflow as tf
from word2vec_basic import DataManager

class DataTest(tf.test.TestCase):
    def setUp(self):
        pass
    def testDataModule(self):
        mgr = DataManager()

        print('Most common words (+UNK)', mgr.count[:5])
        print('Sample data', mgr.data[:10], [mgr.reverse_dictionary[i] for i in mgr.data[:10]])

        num_steps = 100
        for _ in range(num_steps):
            batch, labels = mgr.generate_batch()
            for i in batch:
                if i > 5000:
                    print(i)
            # mgr.print_batch(batch, labels)

if __name__ == '__main__':
    tf.test.main()
