import numpy as np
from argparse import ArgumentParser


class Encoding:
    """
    Encodes/Decodes the data into/from integers that will be efed to the RNN.
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
