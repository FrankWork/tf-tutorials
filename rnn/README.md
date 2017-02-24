# RNN
This directory contains functions for creating recurrent neural networks
and sequence-to-sequence models. Detailed instructions on how to get started
and use them are available in the tutorials.

* [RNN Tutorial](http://tensorflow.org/tutorials/recurrent/)
* [Sequence-to-Sequence Tutorial](http://tensorflow.org/tutorials/seq2seq/)

Here is a short overview of what is in this directory.

File | What's in it?
--- | ---
`ptb/` | PTB language model, see the [RNN Tutorial](http://tensorflow.org/tutorials/recurrent/)
`translate/` | Translation model, see the [Sequence-to-Sequence Tutorial](http://tensorflow.org/tutorials/seq2seq/)

## RNN-LSTM Language Model

- [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Zaremba et al., 2014](http://arxiv.org/abs/1409.2329)

PTB dataset from Tomas Mikolov's webpage:
```bash
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
```

Run the code:
```bash
python ptb_word_lm.py --data_path=simple-examples/data/ --model=small
```

##  Sequence-to-Sequence Models

- [Cho et al., 2014](http://arxiv.org/abs/1406.1078) : Basic Seq-to-Seq Model
- [Sutskever et al., 2014](https://arxiv.org/abs/1409.3215): Multi-layer
- [Bahdanau et al., 2014](http://arxiv.org/abs/1409.0473): Attention Mechanism Model
- [Jean et. al., 2014](https://arxiv.org/abs/1412.2007): Very Large Target Vocabulary


Download data from the [WMT'15 Website](http://www.statmt.org/wmt15/translation-task.html).  
It takes about 20GB of disk space.
```bash
cd translate/
python translate.py --data_dir [your_data_directory]
```
