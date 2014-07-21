#coding: utf-8

import random, numpy
from function import sigmoid, sigmoid_prime
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


    def fix_traning_layer(self):
        self.training_layer += 1
        if self.training_layer == len(self.num_nodes) - 1:
            self.weights = []
            for ae in self.auto_encoders:
                self.weights.append(numpy.copy(ae.weight))
            # TODO: initialize weight with good defaults.
            # 0-1 classification
            self.weights.append(numpy.array([[1.0 - random.random()*2 for x in xrange(self.num_nodes[-1])]]))

            print "create prediction layer"

    def fine_tuning(self, features, labels, alpha = 0.1):
        num_weights = len(self.weights)
        for x, z in zip(features, labels):
            outputs = [numpy.array([x]).transpose()]
            for w in self.weights:
                # TODO: other activate function
                outputs.append(sigmoid(numpy.dot(w, outputs[-1])))

            # calc. delta
            deltas = []
            for i in xrange(num_weights, 0, -1):
                if i is num_weights:
                    # top level
                    z = numpy.array([z])
                    deltas.append(z-outputs[i])
                else:
                    deltas.append(numpy.multiply(sigmoid_prime(outputs[i]).transpose(), numpy.dot(deltas[-1], self.weights[i])))
            deltas.reverse()

            # update weight
            for i in xrange(num_weights):
                self.weights[i] += alpha * numpy.dot(outputs[i], deltas[i]).transpose()

    def predict(self, x):
        outputs = [numpy.array([x]).transpose()]
        for w in self.weights:
            # TODO: other activate function
            outputs.append(sigmoid(numpy.dot(w, outputs[-1])))
        return outputs[-1]





if __name__=='__main__':
    # 0-1 dataset
    features, labels = get_binary_dataset()

    N = len(features)
    V = len(features[0])


    sae = StackedAutoEncoder(V, [V+2,V+4])

    indices = range(N)
    random.shuffle(indices)
    print len(features)
    train_features = numpy.take(features, indices[:3*N/4], axis=0).tolist()
    train_labels   = numpy.take(labels, indices[:3*N/4], axis=0).tolist()
    test_features = numpy.take(features, indices[3*N/4:], axis=0).tolist()
    test_labels   = numpy.take(labels, indices[3*N/4:], axis=0).tolist()
    print len(train_features)


    print "---layer1---"
    for i in xrange(10000):
        j = int(random.random()*3*(N-10)/4)
        sae.train(train_features[j:j+10])
        if i<100 or i%1000 == 0:
            print sae.error(train_features)

    sae.fix_traning_layer()

    print "---layer2---"
    for i in xrange(10000):
        j = int(random.random()*3*(N-10)/4)
        sae.train(train_features[j:j+10])
        if i%1000 == 0:
            print sae.error(train_features)

    sae.fix_traning_layer()

    print "---fine-tuning---"
    for i in xrange(5000):
        j = int(random.random()*3*(N-10)/4)
        sae.fine_tuning(train_features[j:j+10], train_labels[j:j+10])

    correct = 0
    for x,z in zip(test_features, test_labels):
        z_prime = sae.predict(x)
        z_prime = 1 if z_prime[0][0] >= 0.5 else 0
        if z_prime is int(z[0]): correct += 1
        print int(z[0]), z_prime
    print 100.0 * correct / len(test_features)






