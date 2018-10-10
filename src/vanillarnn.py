import numpy as np
from argparse import ArgumentParser


class Encoding:
    """
    Encodes/Decodes the data into/from integers that will be efed to the RNN.
    Based on the gist of Andrej Karpathy https://gist.github.com/karpathy/d4dee566867f8291f086
    """

    def __init__(self, vocabulary):
        self._char_to_int = {}
        self._int_to_char = {}
        for i, ch in enumerate(vocabulary):
            self._char_to_int[ch] = i
            self._int_to_char[i] = ch

    def encode(self, sequence):
        return [self._char_to_int[ch] for ch in sequence]

    def decode(self, sequence):
        return "".join([self._int_to_char[i] for i in sequence])


class VanillaRNN:
    """
    A simple implementation of Recurrent Neural Network.
    """

    def __init__(self, vocabulary_size, hidden_size, sequence_length, encoding):
        self._vocab_size = vocabylary_size
        self._hidden_size = hidden_size
        self._sequence_length = sequence_length
        self._encoding = encoding

        self._initialize_model_parameters()
        self._initialize_model_memory()

        self._smooth_loss = -np.log(1.0 / self._vocab_size) * self._sequence_length
        self._hidden = np.zeros((self._hidden_size, 1))

    def _initialize_model_memory(self):
        self._mWxh = np.zeros_like(self._Wxh)
        self._mWhh = np.zeros_like(self._Whh)
        self._mWhy = np.zeros_like(self._Why)
        self._mbh = np.zeros_like(self._bh)
        self._mby = np.zeros_like(self._by)

    def _initialize_model_parameters(self):
        # Input to hidden parameters
        self._Wxh = np.random.randn(self._hidden_size, self._vocab_size) * .01
        # Hidden to hidden parameters
        self._Whh = np.random.randn(self._hidden_size, self._hidden_size) * .01
        # Hidden to output parameters
        self._Why = np.random.randn(self._vocab_size, self._hidden_size) * .01
        # Hidden bias
        self._bh = np.zeros((self._hidden_size, 1))
        # Output bias
        self._by = np.zeros((self._vocab_size, 1))

    def _sample(self, seed_char, sample_size):
        return 0

    def fit(self, data, num_epochs=500, sample_interval=100, sample_size=200):
        p = 0
        for epoch in range(num_epochs):
            if p + self._sequence_length + 1 >= len(data):
                continue
            inputs = self._encoding.encode(data[p:p + self._sequence_length])
            labels = self._encoding.encode(data[p + 1:p + self._sequence_length + 1:])
            if epoch % sample_interval == 0:
                sample = self._sample(inputs[0], sample_size)
                print('-----\n{}\n-----'.format(self._encoding.decode(sample)))


def read_data(input_file):
    data = open(input_file).read()
    return data


def run(args):
    data = read_data(args.corpus_file)
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print("Data has {} characters; {} unique.".format(data_size, vocab_size))
    e = Encoding(chars)
    print(e.decode(e.encode('manelator')))


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--corpus-file',
                        required=True,
                        help='The name of the file containing training corpus.')
    parser.add_argument('--hidden-size',
                        required=False,
                        type=int,
                        default=100,
                        help='The size of hidden layer of neurons.')
    parser.add_argument('--sequence-length',
                        required=False,
                        type=int,
                        default=25,
                        help='Number of steps to unroll the RNN for.')
    parser.add_argument('--learning-rate',
                        required=False,
                        type=float,
                        default=1e-1,
                        help='Learning rate.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
