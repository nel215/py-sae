#coding: utf-8
import requests
import os.path
import pickle

def get_binary_dataset():
    # 0-1 dataset
    dataset = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train').text
    dataset = map(lambda row: row.split(','), dataset.split('\n'))
    titles = dataset[0]
    dataset = dataset[1:]
    dataset = filter(lambda data: len(data) > 1, dataset)
    features = map(lambda data: map(float, data[:-1]), dataset)
    labels = map(lambda data: map(float, data[-1:]), dataset)

    return (features, labels)

def get_mushroom_dataset():
    filename = './tmp/mushroom.dat'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if os.path.isfile(filename):
        f = open(filename, 'r')
        return pickle.load(f)
    dataset = requests.get('http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms').text
    num_feature = 112
    features = []
    labels = []
    dataset = filter(lambda data: len(data)>1, dataset.split('\n'))
    for data in dataset:
        data = data.split(' ')
        labels.append(1 if data[0] == '2' else 0)
        feature = [0 for f in xrange(num_feature)]
        for [bin, _] in map(lambda d: d.split(':'), filter(lambda d: len(d)>1, data[1:])):
            feature[int(bin)-1] = 1
        features.append(feature)
    result = (features, labels)
    f = open(filename, 'w')
    pickle.dump(result, f)
    f.close()
    return result

if __name__=='__main__':
    get_mushroom_dataset()



