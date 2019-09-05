#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMBHN (Inductive Model based on Bipartite Heterogeneous Network - Regression)
=====================================================

Copyright (C) 2016 Alan Valejo <alanvalejo@gmail.com> All rights reserved
Copyright (C) 2016 Thiago Faleiros <thiagodepaulo@gmail.com> All rights reserved

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

.. original java implementation:
.. http://sites.labic.icmc.usp.br/thiagopf/prl_2016/text_categorization_tool/TCTAlgorithms/InductiveSupervised/IMBHN_R_InductiveSupervised.java

"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

__maintainer__ = 'Thiago de Paulo Faleiros and Alan Valejo'
__author__ = 'Alan Valejo, Thiago Faleiros'
__email__ = 'alanvalejo@gmail.com', 'thiagodepaulo@gmail.com'
__credits__ = ['Alan Valejo', 'Thiago Faleiros']
__homepage__ = 'https://www.alanvalejo.com.br/software?name=pbg'
__license__ = 'GNU.GPL.v3'
__docformat__ = 'markdown en'
__version__ = '2.1'
__date__ = '2019-09-01'

class IMBHN(BaseEstimator, ClassifierMixin):

    def __init__(self, **kwargs):

        prop_defaults = {'eta': 0.1, 'max_itr': 100, 'min_sqr_error': 1e-6}

        self.__dict__.update(prop_defaults)
        self.__dict__.update(kwargs)

    def stop_analysis(self, mean_error, num_iterations):

        if mean_error - self.mean_error == 0:
            return True
        if mean_error < self.min_sqr_error:
            return True
        if num_iterations >= self.max_itr:
            return True
        self.mean_error = mean_error
        self.niterations = num_iterations
        return False

    def init_dataset(self, X, y):

        self.ndocs, self.nterms = X.shape
        self.nclass = len(set(y))
        self.W = X # document-term matrix
        self.D = range(self.ndocs) # set of documents
        self.C = range(self.nclass) # set of classes
        self.T = range(self.nterms) # set of terms
        self.Y = y # class labels
        self.current_doc_index = -1
        self.terms_by_doc = []
        self.small_float = 0.000001
        self.mean_error = float("-inf")

    def get_terms_by_doc(self, d):

        if self.current_doc_index == d:
            return self.terms_by_doc
        self.current_doc_index = d
        self.terms_by_doc = self.W.indices[self.W.indptr[d]:self.W.indptr[d+1]]
        return self.terms_by_doc

    def classify(self, d):

        out = np.zeros(self.nclass)
        for c in self.C:
            cw = 0 # class_weight
            for t in self.set_of_terms_by_doc(d):
                cw += self.F[t, c] * self.W[d, t]
            out[c] = cw
        return out

    def classify_hard(self, d):

        _max, _max_c = float("-inf"), -1
        for c in self.C:
            cw = 0 # class_weight
            for t in self.get_terms_by_doc(d):
                cw += self.F[t, c] * self.W[d, t]
            if _max < cw:
                _max, _max_c = cw, c
        out = np.zeros(self.nclass)
        if _max > self.small_float:
            out[_max_c] = 1
        return out

    def fit(self, X, y):

        self.init_dataset(X, y)
        _exit = False
        self.F = np.zeros((self.nterms, self.nclass))
        self.num_itr = 0

        while _exit is False:
            mean_error = 0.0
            for d in self.D:
                estimated_classes = self.classify_hard(d)
                for c in self.C:
                    error = (1 if self.Y[d] == c else 0) - estimated_classes[c]
                    mean_error += ((error * error) / 2.0)
                    for t in self.get_terms_by_doc(d):
                        current_weight = self.F[t, c]
                        new_weight = current_weight + (self.eta * self.W[d, t] * error)
                        self.F[t, c] = new_weight
            self.num_itr += 1
            mean_error = mean_error / self.ndocs
            _exit = self.stop_analysis(mean_error, self.num_itr)

        return self

    def predict(self, X):

        ndocs = X.shape[0]
        result = np.zeros(ndocs)
        for d in range(ndocs):
            _max, _max_c = float("-inf"), -1
            for c in self.C:
                cw = 0 # class_weight
                for t in X.indices[X.indptr[d]: X.indptr[d + 1]]:
                    cw += self.F[t, c] * X[d, t]
                if _max < cw:
                    _max, _max_c = cw, c
            result[d] = _max_c
        return result
