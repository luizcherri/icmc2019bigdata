#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import pandas
import util

from pbg import PBG
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score

datasets = ['input/synthetic/document_term_easy', 'input/synthetic/document_term_hard']

for dataset in datasets:

    print('Dataset:', dataset.split('_')[-1].title())

    membership = numpy.loadtxt(dataset + '.membership', skiprows=0, dtype=int)
    types = numpy.loadtxt(dataset + '.type', skiprows=0, dtype=int)
    data = pandas.read_csv(dataset + '.ncol', header=None, sep=' ')

    counter = Counter(types)
    vertices = list(counter.values())
    type_keys = list(counter.keys())

    weights = numpy.array(data[2])
    edges = data[[0, 1]].values
    row, column = map(numpy.array, zip(*edges))
    column = column - vertices[0]
    X = csr_matrix((weights, (row, column)), shape=(vertices[0], vertices[1]))
    y = membership[:vertices[0]]
    k = len(numpy.unique(y))

    # Init labels
    number_labelled_examples = [10, 20, 30, 40, 50, 100]
    conf_labels = util.ConfigLabels(list_n_labels=number_labelled_examples)
    conf_labels.fit(y)

    accuracy = []
    for labelled in number_labelled_examples:
        y_semi = conf_labels.semi_labels[labelled]
        model = PBG(n_components=k)
        model = model.fit(X, y_semi)
        y_unlabelled = conf_labels.unlabelled_idx[labelled]
        predictions = model.transduction_[y_unlabelled]
        y_test = numpy.array(y)[y_unlabelled]
        accuracy.append(accuracy_score(y_test, predictions))

    accuracy = ['%.2f' % i for i in accuracy]
    result = dict(zip(number_labelled_examples, accuracy))

    print('Result:', result)

exit()
