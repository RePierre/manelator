
class Encoding:
    """
    Encodes/Decodes the data into/from integers that will be efed to the RNN.
    Based on the gist of Andrej Karpathy
    https://gist.github.com/karpathy/d4dee566867f8291f086
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
