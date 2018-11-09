import numpy as np

# data I/O
data = open('scrapper.py', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has {} characters, {} unique.'.format(data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

np.random.seed(0)


class VanillaRNN:
    """Vanilla RNN
    """

    def __init__(self,
                 hidden_size=100,
                 sequence_length=25,
                 learning_rate=1e-1):

        self._hidden_size = hidden_size
        self._sequence_length = sequence_length
        self._learning_rate = learning_rate

        # model parameters
        self._Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
        self._Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self._Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
        self._bh = np.zeros((hidden_size, 1))  # hidden bias
        self._by = np.zeros((vocab_size, 1))  # output bias

    def lossFun(self, inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self._Wxh, xs[t]) + np.dot(self._Whh, hs[t - 1]) + self._bh)  # hidden state
            ys[t] = np.dot(self._Why, hs[t]) + self._by  # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
            loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self._Wxh), np.zeros_like(self._Whh), np.zeros_like(self._Why)
        dbh, dby = np.zeros_like(self._bh), np.zeros_like(self._by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self._Why.T, dy) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self._Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

    def sample(self, h, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self._Wxh, x) + np.dot(self._Whh, h) + self._bh)
            y = np.dot(self._Why, h) + self._by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(vocab_size), p=p.ravel())
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def fit(self):
        n, p = 0, 0
        mWxh, mWhh, mWhy = np.zeros_like(self._Wxh),\
            np.zeros_like(self._Whh),\
            np.zeros_like(self._Why)
        # memory variables for Adagrad
        mbh, mby = np.zeros_like(self._bh), np.zeros_like(self._by)
        # loss at iteration 0
        smooth_loss = -np.log(1.0 / vocab_size) * self._sequence_length
        while True:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + self._sequence_length + 1 >= len(data) or n == 0:
                hprev = np.zeros((self._hidden_size, 1))  # reset RNN memory
                p = 0  # go from start of data
            inputs = [char_to_ix[ch] for ch in data[p:p + self._sequence_length]]
            targets = [char_to_ix[ch] for ch in data[p + 1:p + self._sequence_length + 1]]

            # sample from the model now and then
            if n % 100 == 0:
                sample_ix = self.sample(hprev, inputs[0], 200)
                txt = ''.join(ix_to_char[ix] for ix in sample_ix)
                print('----\n {} \n----'.format(txt))

            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if n % 100 == 0:
                print('iter {}, loss: {}'.format(n, smooth_loss))  # print progress

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self._Wxh, self._Whh, self._Why, self._bh, self._by],
                                          [dWxh, dWhh, dWhy, dbh, dby],
                                          [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -self._learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            p += self._sequence_length  # move data pointer
            n += 1  # iteration counter


if __name__ == '__main__':
    rnn = VanillaRNN()
    rnn.fit()
