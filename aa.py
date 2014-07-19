#coding: utf-8
import random, numpy
import requests

class AutoEncoder:
    def __init__(self, V, H):
        # TODO: add constant term.
        self.V = V
        self.H = H
        # TODO: initialize weight with good defaults.
        self.weight = numpy.matrix([[1.0 - random.random()*2 for x in xrange(self.V)] for y in xrange(self.H)])

    def train(self, samples, alpha = 0.05):
        delta = numpy.matrix([[0.0 for x in xrange(self.V)] for y in xrange(self.H)]).transpose()
        for x in samples:
          y = self.encode(x)
          z = self.decode(y)

          sum = numpy.dot(self.weight, x-z)
          delta += numpy.dot((x-z), y.transpose())
          delta += numpy.dot(x, numpy.multiply(numpy.multiply(y, 1.0-y), sum).transpose())

        self.weight += alpha * delta.transpose() / len(samples)

    def encode(self, x):
        h1 = numpy.dot(self.weight, x)
        return 1.0 / (1.0 + numpy.exp(-h1))

    def decode(self, y):
        h2 = numpy.dot(self.weight.transpose(), y)
        return 1.0 / (1.0 + numpy.exp(-h2))

    def error(self, samples):
        error = numpy.matrix([[0.0] for x in xrange(self.V)])
        for x in samples:
            y = self.encode(x)
            z = self.decode(y)
            error += numpy.multiply(x, numpy.log(z)) - numpy.multiply((1.0 - x), numpy.log(1.0 - z))
        return numpy.sum(error)



if __name__=='__main__':
    # 0-1 dataset
    resp = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train')
    samples = map(lambda row: row.split(','), resp.text.split('\n'))
    titles = samples[0]
    samples = samples[1:]
    samples = filter(lambda arr: len(arr) > 1, samples)
    samples = map(lambda arr: numpy.matrix([map(float, arr)]), samples)
    samples = map(lambda mat: mat.transpose(), samples)

    V = samples[0].shape[0]
    H = 2*V
    print V,H

    aa = AutoEncoder(V,H)

    for i in xrange(5000):
        j = int(random.random()*len(samples))
        #print samples[j:j+10]
        aa.train(samples[j:j+10])
        if i<100 or i%1000 == 0:
            print aa.error(samples)

    for sample in samples:
        print numpy.abs(sample - aa.decode(aa.encode(sample)))



