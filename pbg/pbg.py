#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PBG (Propagation in Bipartite Graph)
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
"""

import numpy as np
import time
import os

from util import RandMatrices
from sklearn.base import BaseEstimator, ClassifierMixin
from os.path import join

__maintainer__ = 'Thiago de Paulo Faleiros and Alan Valejo'
__author__ = 'Alan Valejo, Thiago Faleiros'
__email__ = 'alanvalejo@gmail.com', 'thiagodepaulo@gmail.com'
__credits__ = ['Alan Valejo', 'Thiago Faleiros']
__homepage__ = 'https://www.alanvalejo.com.br/software?name=pbg'
__license__ = 'GNU.GPL.v3'
__docformat__ = 'markdown en'
__version__ = '2.1'
__date__ = '2019-09-01'

class PBG(BaseEstimator, ClassifierMixin):

    def __init__(self, **kwargs):

        prop_defaults = {'n_components': None, 'alpha': 0.05,
         'beta': 0.0001, 'local_max_itr': 50, 'global_max_itr': 10,
         'local_threshold': 1e-6, 'global_threshold': 1e-6,
         'max_time': 18000, 'save_interval': -1, 'dir': 'output',
         'calc_q': False, 'debug': False,
         'rand_init': False}

        self.__dict__.update(prop_defaults)
        self.__dict__.update(kwargs)

    def normalizedbycolumn_map(self, B):

        n = len(B.values()[0])
        col_sum = np.zeros(n)
        for key in B:
            vet = B[key]
            for i in range(n):
                col_sum[i] += vet[i]
        for key in B:
            vet = B[key]
            for i in range(n):
                vet[i] /= col_sum[i]
                vet[i] = self.beta + vet[i]
        return B

    def normalizebycolumn_plus_beta(self, B):

        if isinstance(B, dict):
            return self.normalizedbycolumn_map(B)
        nrow, ncol = B.shape
        for i in range(ncol):
            B[:, i] /= B[:, i].sum()
        return self.beta + B

    def Q2(self, X, D, A, B, alpha):

        CONST = 0.0000001
        _sum = 0
        for d_j in D:
            for w_i, f_ji in zip(X.indices[X.indptr[d_j]:X.indptr[d_j+1]], X.data[X.indptr[d_j]:X.indptr[d_j+1]]):
                AB_ji = A[d_j]*B[w_i]
                C_ji = (AB_ji / AB_ji.sum())
                _sum += sum((f_ji * C_ji) * (np.log((AB_ji + CONST) / (C_ji + CONST))))
            _sum -= sum((alpha - A[d_j]) * np.log(A[d_j] + CONST) - A[d_j] * (np.log(A[d_j] + CONST) - 1))
        return _sum

    def Q(self, X, D, A, B):

        _sum = 0
        for d_j in D:
            for w_i, f_ji in zip(X.indices[X.indptr[d_j]:X.indptr[d_j+1]], X.data[X.indptr[d_j]:X.indptr[d_j+1]]):
                sumAjBi = sum(A[d_j]*B[w_i])
                _sum += f_ji * np.log(f_ji / (sumAjBi)) - f_ji + (sumAjBi)
        return _sum

    def global_propag(self, Xcrc, W, A, B):

        for w_i in W:
            nB_i = np.zeros(self.n_components)
            for d_j, f_ji in zip(Xcrc.indices[Xcrc.indptr[w_i]:Xcrc.indptr[w_i+1]], Xcrc.data[Xcrc.indptr[w_i]:Xcrc.indptr[w_i+1]]):
                H = (A[d_j] * B[w_i])
                nB_i += f_ji * (H / H.sum())
            B[w_i] = nB_i
        return self.normalizebycolumn_plus_beta(B)

    def local_propag(self, X, d_j, A_j, B):

        nA_j = np.zeros(len(A_j))
        for w_i, f_ji in zip(X.indices[X.indptr[d_j]:X.indptr[d_j+1]], X.data[X.indptr[d_j]:X.indptr[d_j+1]]):
            H = (A_j * B[w_i])
            nA_j += f_ji * (H / H.sum())
        nA_j += self.alpha
        return nA_j

    def suppress(self, A_j, y_j):

        aux = A_j[y_j]
        A_j.fill(0)
        A_j[y_j] = aux

    def bgp(self, X, Xcrc, W, D, A, B, labelled=None):

        self.global_niter = 0

        t0 = time.time()
        while self.global_niter <= self.global_max_itr :
            self.global_niter += 1

            if time.time() - t0 > self.max_time:
                break
            if self.save_interval!= None and self.save_interval % self.global_niter == 0 :
                self.save_matrices(A, B, self.global_niter)

            for d_j in D:
                local_niter = 0
                if self.debug and self.global_niter % 10 == 0: print(d_j)
                while local_niter <= self.local_max_itr:
                    local_niter += 1
                    oldA_j = np.array(A[d_j])
                    A[d_j] = self.local_propag(X, d_j, A[d_j], B)
                    mean_change = np.mean(abs(A[d_j] - oldA_j))
                    if mean_change <= self.local_threshold:
                        # if self.debug: print('converged itr %s' %local_niter)
                        break
                if (labelled is not  None) and (labelled[d_j] != -1):
                    self.suppress(A[d_j], labelled[d_j])
            self.global_propag(Xcrc, W, A, B)
            if self.calc_q:
                q = self.Q2(X, D, A, B, self.alpha)
                if self.debug: print('itr %s Q %s' % (self.global_niter, q))

    def fit(self, X, y=None):

        rand = RandMatrices()
        # D: set of documents indices
        # W: set of word indices
        # K: number of topics
        D, W, K = range(X.shape[0]), range(X.shape[1]), self.n_components
        A,B = rand.create_rand_matrices(D ,W ,K ) if self.rand_init else rand.create_label_init_matrices(X, D, W, K, y, self.beta,-1)

        # Convert matriz
        Xcsc = X.tocsc()
        self.bgp(X, Xcsc, W, D, A, B, labelled=y)
        self.components_ = B.transpose()
        if y is not None:
            # Label construction
            # Truct a categorical distribution for classification only
            classes = np.unique(y)
            classes = (classes[classes != -1])
            self.classes_ = classes
            # Assign class indexes to unlabeled examples
            self.transduction_ = self.classes_[np.argmax(A, axis=1)].ravel()
            # Create label distribution (normalizes matrix A)
            normalizer = np.atleast_2d(np.sum(A, axis=1)).T
            self.label_distributions_ = A / normalizer

        return self

    def transform(self, X):
        return None

    def save_matrices(self, A, B, global_niter):

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        np.save(join(self.dir, 'A_' + str(global_niter)), A)
        np.save(join(self.dir, 'B_' + str(global_niter)), B)
