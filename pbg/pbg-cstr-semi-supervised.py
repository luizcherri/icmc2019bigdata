#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import util

from sklearn.metrics import accuracy_score
from pbg import PBG

# Read and create X and y
X, y, K = util.loadarff('input/arff/cstr.arff')

# Init labels
number_labelled_examples = [10, 20, 30, 40, 50, 100]
conf_labels = util.ConfigLabels(list_n_labels=number_labelled_examples)
conf_labels.fit(y)

accuracy = []
for labelled in number_labelled_examples:
    y_semi = conf_labels.semi_labels[labelled]
    model = PBG(n_components=K)
    model = model.fit(X, y_semi)
    y_unlabelled = conf_labels.unlabelled_idx[labelled]
    predictions = model.transduction_[y_unlabelled]
    y_test = numpy.array(y)[y_unlabelled]
    accuracy.append(accuracy_score(y_test, predictions))

accuracy = ['%.2f' % i for i in accuracy]
result = dict(zip(number_labelled_examples, accuracy))
print('Result:', result)
exit()
