#coding: utf-8
import random, numpy
import requests

class AutoEncoder:
    def __init__(self, V, H):
        self.V = V
        self.H = H
        # TODO: あとで良い感じの初期値にする
        self.weight = numpy.matrix([[1.0 - random.random()*2 for x in xrange(self.V)] for y in xrange(self.H)])

    def train(self, x, alpha = 0.05):
        h1 = numpy.dot(self.weight, x)
        y = 1.0 / (1.0 + numpy.exp(-h1))
        h2 = numpy.dot(self.weight.transpose(), y)
        z = 1.0 / (1.0 + numpy.exp(-h2))

        sum = numpy.dot(self.weight, x-z)
        delta1 = numpy.dot((x-z), y.transpose())
        delta2 = numpy.dot(x, numpy.multiply(numpy.multiply(y, 1.0-y), sum).transpose())
        delta = delta1 + delta2

        self.weight += alpha * delta.transpose()

        print numpy.multiply(x, numpy.log(z)) - numpy.multiply((1.0 - x), numpy.log(1.0 - z))

if __name__=='__main__':
    # 0-1のテストデータ
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

    for i in xrange(100):
        for sample in samples:
            aa.train(sample)

