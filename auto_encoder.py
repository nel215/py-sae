#coding: utf-8
from function import sigmoid
from dataset import get_binary_dataset
import random, numpy

class AutoEncoder:
    def __init__(self, V, H):
        # TODO: add bias term.
        self.V = V
        self.H = H
        # TODO: initialize weight with good defaults.
        self.weight = numpy.array([[1.0 - random.random()*2 for x in xrange(self.V)] for y in xrange(self.H)])

    def train(self, samples, alpha = 0.05):
        delta = numpy.array([[0.0 for x in xrange(self.V)] for y in xrange(self.H)]).transpose()
        for x in samples:
            y = self.encode(x)
            z = self.decode(y)

            sum = numpy.dot(self.weight, x-z)
            delta += numpy.dot((x-z), y.transpose())
            delta += numpy.dot(x, numpy.multiply(numpy.multiply(y, 1.0-y), sum).transpose())

        self.weight += alpha * delta.transpose() / len(samples)

    def encode(self, x):
        h1 = numpy.dot(self.weight, x)
        return sigmoid(h1)

    def decode(self, y):
        h2 = numpy.dot(self.weight.transpose(), y)
        return sigmoid(h2)

    def error(self, samples):
        error = numpy.array([[0.0] for x in xrange(self.V)])
        for x in samples:
            y = self.encode(x)
            z = self.decode(y)
            error += numpy.multiply(x, numpy.log(z)) - numpy.multiply((1.0 - x), numpy.log(1.0 - z))
        return numpy.sum(error)



if __name__=='__main__':
    features, labels = get_binary_dataset()

    V = len(features[0])
    H = 2*V

    aa = AutoEncoder(V,H)


    #for i in xrange(10000):
    #    j = int(random.random()*len(samples))
    #    #print samples[j:j+10]
    #    aa.train(samples[j:j+10])
    #    if i<100 or i%1000 == 0:
    #        print aa.error(samples)

    #for sample in samples:
    #    print numpy.abs(sample - aa.decode(aa.encode(sample)))



