import numpy as np
from encoding import Encoding

# data I/O
data = open('scrapper.py', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has {} characters, {} unique.'.format(data_size, vocab_size))

np.random.seed(0)


class VanillaRNN:
    """Vanilla RNN
    """

    def __init__(self,
                 encoding,
                 hidden_size=100,
                 sequence_length=25,
                 learning_rate=1e-1):

        self._encoding = encoding
        self._hidden_size = hidden_size
        self._sequence_length = sequence_length
        self._learning_rate = learning_rate

    def _initialize_model_parameters(self, vocab_size):
        """
        Initializes model parameters.
        Weight matrices are initialized to small random values;
        bias vectors are initialized to zeros.
        """
        # Input to hidden matrix
        self._Wxh = np.random.randn(self._hidden_size, vocab_size) * 0.01
        # Hidden to hidden matrix
        self._Whh = np.random.randn(self._hidden_size, self._hidden_size) * 0.01
        # Hidden to output matrix
        self._Why = np.random.randn(vocab_size, self._hidden_size) * 0.01
        # Hidden layer bias
        self._bh = np.zeros((self._hidden_size, 1))
        # Output layer bias
        self._by = np.zeros((vocab_size, 1))

    def _reset_model_memory(self):
        self._h = np.zeros((self._hidden_size, 1))

    def _initialize_Adagrad_memory(self):
        """
        Initializes memory for Adagrad parameter update.
        Memory structures have the same shape as model parameters
        but are initialized to zeros.
        """
        self._ada_Wxh = np.zeros_like(self._Wxh)
        self._ada_Whh = np.zeros_like(self._Whh)
        self._ada_Why = np.zeros_like(self._Why)
        self._ada_bh = np.zeros_like(self._bh)
        self._ada_by = np.zeros_like(self._by)

    def _apply_loss_function(self, inputs, targets):
        """
        Calculates cross-entropy loss and updates model hidden state.
        `inputs`, `targets` are both list of integers.
        Returns the loss and the gradients on model parameters.
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(self._h)
        loss = 0

        # Forward pass
        for t in range(len(inputs)):
            # One-hot encoding of current input
            xs[t] = np.zeros((vocab_size, 1))
            xs[t][inputs[t]] = 1
            # Compute values for the hidden state
            hs[t] = np.tanh(np.dot(self._Wxh, xs[t]) + np.dot(self._Whh, hs[t - 1]) + self._bh)
            # Calculate unnormalized log probabilities for the next chars
            ys[t] = np.dot(self._Why, hs[t]) + self._by
            # Calculate probabilities fo the next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            # Update cross-entropy (softmax) loss
            loss += -np.log(ps[t][targets[t], 0])

        # Backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self._Wxh), np.zeros_like(self._Whh), np.zeros_like(self._Why)
        dbh, dby = np.zeros_like(self._bh), np.zeros_like(self._by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            # Back-propagate into y by specifying the direction
            # which leads to loss decrease
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            # Back-propagate into h
            dh = np.dot(self._Why.T, dy) + dhnext
            # Back-propagate through tanh nonlinearity
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self._Whh.T, dhraw)

        # Apply clipping to avoid exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        self._h = hs[len(inputs) - 1]
        return loss, dWxh, dWhh, dWhy, dbh, dby

    def _sample_sequence(self, seed_ix, length):
        """
        Sample a sequence of `length` integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        h = np.copy(self._h)
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(length):
            h = np.tanh(np.dot(self._Wxh, x) + np.dot(self._Whh, h) + self._bh)
            y = np.dot(self._Why, h) + self._by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(vocab_size), p=p.ravel())
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def _update_model_parameters(self, dWxh, dWhh, dWhy, dbh, dby):
        """
        Perform  Adagrad update of model parameters.
        """
        for param, dparam, mem in zip([self._Wxh, self._Whh, self._Why, self._bh, self._by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [self._ada_Wxh, self._ada_Whh, self._ada_Why, self._ada_bh, self._ada_by]):
            mem += dparam * dparam
            # Adagrad update
            param += -self._learning_rate * dparam / np.sqrt(mem + 1e-8)

    def fit(self, sample_size=200, sample_frequency=100):
        """
        Fits the model to the training data.

        Parameters:
        ----------
        sample_size: int, optional
            The length of the sampled sequence from the model.
        sample_frequency: int, optional
            The frequency, in number of epochs, in which to sample from the model.
        """
        n, p = 0, 0
        self._initialize_model_parameters(vocab_size)
        self._initialize_Adagrad_memory()
        # loss at iteration 0
        smooth_loss = -np.log(1.0 / vocab_size) * self._sequence_length
        while True:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + self._sequence_length + 1 >= len(data) or n == 0:
                self._reset_model_memory()
                p = 0  # go from start of data

            inputs = data[p:p + self._sequence_length]
            inputs = self._encoding.encode(inputs)
            targets = data[p + 1:p + self._sequence_length + 1]
            targets = self._encoding.encode(targets)

            # sample from the model now and then
            if n % 100 == 0:
                sample_ix = self._sample_sequence(inputs[0], sample_size)
                print('----\n {} \n----'.format(self._encoding.decode(sample_ix)))

            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby = self._apply_loss_function(inputs, targets)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            # Print progress
            if n % 100 == 0:
                print('iter {}, loss: {}'.format(n, smooth_loss))

            self._update_model_parameters(dWxh, dWhh, dWhy, dbh, dby)

            p += self._sequence_length  # move data pointer
            n += 1  # iteration counter


if __name__ == '__main__':
    rnn = VanillaRNN(Encoding(chars))
    rnn.fit()
