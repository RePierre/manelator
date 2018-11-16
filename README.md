# Unrolling Recurrent Neural Networks

This repository contains all the code & documents for the [presentation](https://iasi.ai/meetups/unrolling-recurrent-neural-networks/) held within [Iași AI community](https://iasi.ai/).

## Refereces
  * [Deep Learning Book](http://www.deeplearningbook.org) by Ian Goodfellow, Yoshua Bengio and Aaron Courville
  * [Lecture on Recurrent Neuural Networks](https://youtu.be/cO0a0QYmFm8) held by Andrej Karpathy
  * [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) article written by Andrej Karpathy
  * [Minimal character-level language model with a Vanilla Recurrent Neural Network gist](https://gist.github.com/karpathy/d4dee566867f8291f086/) written by Andrej Karpathy

## The experiment

The experiment part of the project was to see whether the vanilla RNN can be trained to generate lyrics for [manele]( http://en.wikipedia.org/wiki/Manele). These lyrics are low quality and very simplistic.

## The results

The [vanilla RNN](./src/vanillarnn.py) from the presentation was trained with 9 hyperparameter configurations as described in table below. Each training cycle ended at 1.000.000 iterations over a sequence of 25 characters.

| Run Id | Hidden size | Learning rate |      Loss |
|--------|-------------|---------------|-----------|
|      1 |         100 |           0.1 |       N/A |
|      2 |         100 |          0.01 | 44.355101 |
|      3 |         100 |         0.001 | 59.384402 |
|      4 |         250 |           0.1 | 48.762288 |
|      5 |         250 |          0.01 | 37.229775 |
|      6 |         250 |         0.001 | 55.494079 |
|      7 |         500 |           0.1 | 52.587987 |
|      8 |         500 |          0.01 | 27.412387 |
|      9 |         500 |         0.001 | 53.266636 |

As can be seen from results, the model got stuck in a local minimum, the best results being achieved by run `8` with `hidden_size=500` and `learning_rate=0.01`.

If you're still curious you can browse the [files with generated samples for each run](./run-results):
* Run 1: [`ni1000000-hs100-is25-lr0.1-2018-11-13-1546.txt`](./run-results/ni1000000-hs100-is25-lr0.1-2018-11-13-1546.txt)
* Run 2: [`ni1000000-hs100-is25-lr0.01-2018-11-13-1552.txt`](./run-results/ni1000000-hs100-is25-lr0.01-2018-11-13-1552.txt)
* Run 3: [`ni1000000-hs100-is25-lr0.001-2018-11-13-1555.txt`](./run-results/ni1000000-hs100-is25-lr0.001-2018-11-13-1555.txt)
* Run 4: [`ni1000000-hs250-is25-lr0.1-2018-11-14-1217.txt`](./run-results/ni1000000-hs250-is25-lr0.1-2018-11-14-1217.txt)
* Run 5: [`ni1000000-hs250-is25-lr0.01-2018-11-14-1216.txt`](./run-results/ni1000000-hs250-is25-lr0.01-2018-11-14-1216.txt)
* Run 6: [`ni1000000-hs250-is25-lr0.001-2018-11-14-1206.txt`](./run-results/ni1000000-hs250-is25-lr0.001-2018-11-14-1206.txt)
* Run 7: [`ni1000000-hs500-is25-lr0.1-2018-11-15-1326.txt`](./run-results/ni1000000-hs500-is25-lr0.1-2018-11-15-1326.txt)
* Run 8: [`ni1000000-hs500-is25-lr0.01-2018-11-15-1253.txt`](./run-results/ni1000000-hs500-is25-lr0.01-2018-11-15-1253.txt)
* Run 9: [`ni1000000-hs500-is25-lr0.001-2018-11-15-1324.txt`](./run-results/ni1000000-hs500-is25-lr0.001-2018-11-15-1324.txt)
