#coding: utf-8
import requests

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


