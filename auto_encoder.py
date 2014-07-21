#coding: utf-8
from function import sigmoid
from dataset import get_binary_dataset
import random, numpy

class AutoEncoder:
    def __init__(self, V, H):
        self.V = V
        self.H = H
        # TODO: initialize weight with good defaults.
        self.weight = numpy.array([[1.0 - random.random()*2 for x in xrange(self.V)] for y in xrange(self.H)])

    def train(self, features, alpha = 0.05):
        delta = numpy.array([[0.0 for x in xrange(self.V)] for y in xrange(self.H)])
        for x in features:
            # TODO: add bias term.
            x = numpy.array([x]).transpose()
            y = self.encode(x)
            z = self.decode(y)

            sum = numpy.dot(self.weight, x-z)
            delta += numpy.dot((x-z), y.transpose()).transpose()
            delta += numpy.dot(x, numpy.multiply(numpy.multiply(y, 1.0-y), sum).transpose()).transpose()

        self.weight += alpha * delta / len(features)

    def encode(self, x):
        h1 = numpy.dot(self.weight, x)
        return sigmoid(h1)

    def decode(self, y):
        h2 = numpy.dot(self.weight.transpose(), y)
        return sigmoid(h2)

    def error(self, features):
        eps = 1e-9
        error = numpy.array([[0.0] for x in xrange(self.V)])
        for x in features:
            # TODO: add bias term.
            x = numpy.array([x]).transpose()
            y = self.encode(x)
            z = self.decode(y)
            error += numpy.multiply(x, numpy.log(z+eps)) - numpy.multiply((1.0-x), numpy.log(1.0-z+eps))
        return numpy.sum(error)

if __name__=='__main__':
    features, labels = get_binary_dataset()


    N = len(features)

    for i in xrange(N):
        features[i] += [1]
    V = len(features[0])
    H = 2*V

    aa = AutoEncoder(V,H)

    for i in xrange(10000):
        j = int(random.random()*N)
        aa.train(features[j:j+10])
        if i<100 or i%1000 == 0:
            print aa.error(features)

    for x in features:
        print numpy.abs(x - aa.decode(aa.encode(x)))



