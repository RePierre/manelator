import numpy as np


class VanillaRNN:
    """
    A simple implementation of Recurrent Neural Network.
    Based on the gist of Andrej Karpathy https://gist.github.com/karpathy/d4dee566867f8291f086
    """

    def __init__(self, vocabulary_size, hidden_size,
                 sequence_length, learning_rate, encoding):
        self._vocab_size = vocabulary_size
        self._hidden_size = hidden_size
        self._sequence_length = sequence_length
        self._learning_rate = learning_rate
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

    def _sample(self, seed_index, sample_size):
        x = np.zeros((self._vocab_size, 1))
        x[seed_index] = 1
        indices = []
        h = self._hidden
        for t in range(sample_size):
            h = np.tanh(np.dot(self._Wxh, x) + np.dot(self._Whh, h) + self._bh)
            y = np.dot(self._Why, h) + self._by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self._vocab_size), p=p.ravel())
            x = np.zeros((self._vocab_size, 1))
            x[ix] = 1
            indices.append(ix)

        return indices

    def _apply_loss_function(self, inputs, targets):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(self._hidden)
        loss = 0
        # Forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((self._vocab_size, 1))
            xs[t][inputs[t]] = 1
            # Hidden state
            hs[t] = np.tanh(np.dot(self._Wxh, xs[t]) + np.dot(self._Whh, hs[t - 1]) + self._bh)
            # Unnormalized log probabilities for next chars
            ys[t] = np.dot(self._Why, hs[t]) + self._by
            # Probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            # Cross-entropy loss
            loss += -np.log(ps[t][targets[t], 0])
        self._smooth_loss = self._smooth_loss * .999 + loss * .001
        # Backward pass
        dWxh, dWhh, dWhy = np.zeros_like(self._Wxh), np.zeros_like(self._Whh), np.zeros_like(self._Why)
        dbh, dby = np.zeros_like(self._bh), np.zeros_like(self._by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] = -1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            #  Backpropagate into h
            dh = np.dot(self._Why.T, dy) + dhnext
            # Backpropagate through tanh
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self._Whh.T, dhraw)
        # Apply clipping to mitigate exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        return dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

    def _update_parameters(self, dWxh, dWhh, dWhy, dbh, dby, hprev):
        for param, dparam, mem in zip([self._Wxh, self._Whh, self._Why, self._bh, self._by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [self._mWxh, self._mWhh, self._mWhy, self._mbh, self._mby]):
            mem += dparam * dparam
            param += -self._learning_rate * dparam / np.sqrt(mem + 1e-8)
        self._hidden = hprev

    def fit(self, data, num_epochs=500, sample_interval=100, sample_size=200):
        for epoch in range(num_epochs):
            self._hidden = np.zeros((self._hidden_size, 1))
            p = 0
            while p + self._sequence_length + 1 < len(data):
                inputs = self._encoding.encode(data[p:p + self._sequence_length])
                labels = self._encoding.encode(data[p + 1:p + self._sequence_length + 1:])
                dWxh, dWhh, dWhy, dbh, dby, hprev = self._apply_loss_function(inputs, labels)
                self._update_parameters(dWxh, dWhh, dWhy, dbh, dby, hprev)
                p += self._sequence_length

            if epoch % sample_interval == 0:
                sample = self._sample(inputs[0], sample_size)
                print('-----\n{}\n-----'.format(self._encoding.decode(sample)))
            print('Epoch {}, loss: {}'.format(epoch, self._smooth_loss))
