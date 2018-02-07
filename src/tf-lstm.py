import tensorflow as tf

flags = tf.flags
flags.DEFINE_string('input', None,
                    'The path to input file.')

FLAGS = flags.FLAGS


class Input():
    """Input data for LSTM network

    """

    def __init__(self):
        self._data = None
        self._chars = None
        self._char_to_idx = None
        self._idx_to_char = None

    @property
    def data(self):
        return self._data

    @property
    def chars(self):
        return self._chars

    @property
    def char_to_index(self):
        return self._char_to_idx

    @property
    def index_to_char(self):
        return self._idx_to_char

    @property
    def data_size(self):
        return len(self.data)

    @property
    def vocab_size(self):
        return len(self.chars)

    def load_from(self, file_name):
        with open(file_name, 'rt') as file:
            self._data = file.read()
        self._chars = list(set(self._data))
        self._char_to_idx = {ch: idx for idx, ch in enumerate(self._chars)}
        self._idx_to_char = {idx: ch for idx, ch in enumerate(self._chars)}


def main(_):
    pass


if __name__ == '__main__':
    tf.app.run()
