#coding: utf-8

import numpy

def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))

def sigmoid_prime(x):
    t = sigmoid(x)
    return numpy.multiply(t, (1.0 - t))
