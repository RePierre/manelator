import datetime
import numpy as np
from argparse import ArgumentParser
from encoding import Encoding
from vanillarnn import VanillaRNN

np.random.seed(2018)


def read_data(input_file):
    data = open(input_file).read()
    return data


def get_output_file(args):
    return 'ni{}-hs{}-is{}-lr{}-{}.txt'.format(
        args.num_iterations,
        args.hidden_size,
        args.sequence_length,
        args.learning_rate,
        datetime.datetime.now().strftime('%Y-%m-%d-%H%M'))


def run(args):
    data = read_data(args.corpus_file)
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print("Data has {} characters; {} unique.".format(data_size, vocab_size))
    e = Encoding(chars)
    model = VanillaRNN(encoding=e, input_size=vocab_size,
                       hidden_size=args.hidden_size,
                       sequence_length=args.sequence_length,
                       learning_rate=args.learning_rate)
    model.fit(data, num_iterations=args.num_iterations)
    with open(get_output_file(args), 'w') as f:
        for _ in range(args.num_samples):
            seed = np.random.randint(low=0, high=vocab_size)
            seq = model.generate_sequence(seed, args.sample_size)
            f.write(e.decode(seq))
            f.write('\n\n')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--corpus-file',
                        required=True,
                        help='The name of the file containing training corpus.')
    parser.add_argument('--num-iterations',
                        required=False,
                        type=int,
                        default=500000,
                        help='Number of training iterations.')
    parser.add_argument('--num-samples',
                        required=False,
                        default=10,
                        type=int,
                        help='Number of text samples to generate.')
    parser.add_argument('--sample-size',
                        required=False,
                        type=int,
                        default=200,
                        help='The length of a text sample.')
    parser.add_argument('--sample-frequency',
                        required=False,
                        type=int,
                        default=100,
                        help='The frequency, in number of epochs, in which to sample from the model during training.')
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
