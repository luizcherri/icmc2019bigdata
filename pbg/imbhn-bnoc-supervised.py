#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import pandas

from imbhn import IMBHN
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
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

    kf = KFold(n_splits=10, random_state=None, shuffle=True)
    accuracy = []
    for index, values in enumerate(kf.split(X)):
        train_index, test_index = values
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = IMBHN()
        model = clf.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy.append(accuracy_score(y_test, predictions))

    avg = sum(accuracy) / float(len(accuracy))
    std = numpy.std(accuracy)
    print('Kfold result:', ['%.2f' % i for i in accuracy])
    print('Avg:', '%.2f' % avg)
    print('Std:', '%.2f' % std)

exit()
