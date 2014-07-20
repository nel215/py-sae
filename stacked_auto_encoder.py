#coding: utf-8

import random, numpy
from function import sigmoid
from auto_encoder import AutoEncoder
from dataset import get_binary_dataset


class StackedAutoEncoder:
    def __init__(self, visible, hiddens):
        # TODO: fine-tuning layer
        self.num_nodes= [visible] + hiddens
        self.auto_encoders = []
        for i in xrange(len(self.num_nodes)-1):
            self.auto_encoders.append(AutoEncoder(self.num_nodes[i], self.num_nodes[i+1]))
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
        if self.training_layer == len(self.num_nodes) - 1:
            self.weights = []
            for ae in self.auto_encoders:
                self.weights.append(numpy.copy(ae.weight))
            # TODO: initialize weight with good defaults.
            # 0-1 classification
            self.weights.append([[1.0 - random.random()*2 for x in xrange(self.num_nodes[-1])]])

            print "create prediction layer"

    def fine_tuning(self, samples, classes):
        for x in samples:
            outputs = [x]
            for w in self.weights:
                # TODO: other activate function
                outputs.append(sigmoid(numpy.dot(w, outputs[-1])))
            print outputs







if __name__=='__main__':
    # 0-1 dataset
    features, labels = get_binary_dataset()

    V = len(features[0])
    H = 2*V

    sae = StackedAutoEncoder(V, [V+2,V])

    #for i in xrange(1000):
    #    j = int(random.random()*len(samples))
    #    #print samples[j:j+10]
    #    sae.train(samples[j:j+10])
    #    if i<100 or i%1000 == 0:
    #        print sae.error(samples)

    #sae.fix_traning_layer()

    #for i in xrange(1000):
    #    j = int(random.random()*len(samples))
    #    #print samples[j:j+10]
    #    sae.train(samples[j:j+10])
    #    if i<100 or i%1000 == 0:
    #        print sae.error(samples)

    #for sample in samples:
    #    print sae.output(sample)

    #sae.fix_traning_layer()

    #for i in xrange(1000):
    #    j = int(random.random()*len(samples))
    #    #print samples[j:j+10]
    #    sae.fine_tuning(samples[j:j+10], numpy.array([[1]])





