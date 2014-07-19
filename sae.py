#coding: utf-8
import requests
import random, numpy
from aa import AutoEncoder


class StackedAutoEncoder:
    def __init__(self, visible, hiddens):
        # TODO: fine-tuning layer
        num_of_nodes= [visible] + hiddens
        self.auto_encoders = []
        for i in xrange(len(num_of_nodes)-1):
            self.auto_encoders.append(AutoEncoder(num_of_nodes[i], num_of_nodes[i+1]))
        self.training_layer = 0

    def train(self, samples, alpha=0.05):
        for i in xrange(self.training_layer):
            samples = map(self.auto_encoders[i].encode, samples)
        self.auto_encoders[self.training_layer].train(samples,alpha)

    def error(self, samples, alpha=0.05):
        for i in xrange(self.training_layer):
            samples = map(self.auto_encoders[i].encode, samples)
        return self.auto_encoders[self.training_layer].error(samples)

    def output(self, sample):
        for i in xrange(self.training_layer):
            sample = self.auto_encoders[i].encode(sample)
        top = self.auto_encoders[self.training_layer]
        return top.decode(top.encode(sample))

    def fix_traning_layer(self):
        self.training_layer += 1




if __name__=='__main__':
    resp = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train')
    samples = map(lambda row: row.split(','), resp.text.split('\n'))
    titles = samples[0]
    samples = samples[1:]
    samples = filter(lambda arr: len(arr) > 1, samples)
    samples = map(lambda arr: numpy.matrix([map(float, arr)]), samples)
    samples = map(lambda mat: mat.transpose(), samples)

    V = samples[0].shape[0]
    H = 2*V

    sae = StackedAutoEncoder(V, [V+2,V])

    for i in xrange(1000):
        j = int(random.random()*len(samples))
        #print samples[j:j+10]
        sae.train(samples[j:j+10])
        if i<100 or i%1000 == 0:
            print sae.error(samples)

    sae.fix_traning_layer()

    for i in xrange(1000):
        j = int(random.random()*len(samples))
        #print samples[j:j+10]
        sae.train(samples[j:j+10])
        if i<100 or i%1000 == 0:
            print sae.error(samples)

    for sample in samples:
        print sae.output(sample)


