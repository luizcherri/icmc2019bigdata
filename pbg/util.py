#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Util is part of PBG (Propagation in Bipartite Graph)
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

import numpy
import glob
import os.path
import codecs
import os
import re
import nltk
import logging

from unidecode import unidecode
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from sklearn.base import TransformerMixin, BaseEstimator
from collections import defaultdict
from scipy import sparse
from scipy.io import arff

__maintainer__ = 'Thiago de Paulo Faleiros and Alan Valejo'
__author__ = 'Alan Valejo, Thiago Faleiros'
__email__ = 'alanvalejo@gmail.com', 'thiagodepaulo@gmail.com'
__credits__ = ['Alan Valejo', 'Thiago Faleiros']
__homepage__ = 'https://www.alanvalejo.com.br/software?name=pbg'
__license__ = 'GNU.GPL.v3'
__docformat__ = 'markdown en'
__version__ = '2.1'
__date__ = '2019-09-01'

def loadarff(filename, class_column=True):

	# Read and create X and y
	data, meta = arff.loadarff(filename)
	corpus_ = data[meta.names()[:-1]] # Everything but the last column
	corpus = corpus_.copy()
	corpus = numpy.asarray(corpus.tolist(), dtype=numpy.float32)
	X = sparse.csr_matrix(corpus)
	y, K = [None] * 2
	if class_column:
		y = data[meta.names()[-1]] # Everything but the last column
		y = encode_categorical(y) # or y = pd.Series(y).astype('category').cat.codes.values
		K = len(numpy.unique(y))
	return X, y, K

def encode_categorical(array):
	d = {key: value for (key, value) in zip(numpy.unique(array), numpy.arange(len(array)))}
	shape = array.shape
	array = array.ravel()
	new_array = numpy.zeros(array.shape, dtype=numpy.int)
	for i in range(len(array)):
		new_array[i] = d[array[i]]
	return new_array.reshape(shape)

class Preprocessor(TransformerMixin, BaseEstimator):

	def __init__(self, lang='english', stop_words=True, stem=True):
		self.lang = lang
		self.stop_words = stop_words
		self.stem = stem

	def cleaning(self, corpus):
		return [self.strip_accents_nonalpha(text) for text in corpus]

	# Remove accents and numeric characteres
	def strip_accents_nonalpha(self, text):
		text = text.lower()
		# if not isinstance(text, unicode):
		# 	text = unicode(text, 'utf-8')
		t = unidecode(text)
		# Works fine, because all non-ASCII from s are replaced with their equivalents
		t.encode("ascii")
		t = re.sub(r'[^a-z]', ' ', t)
		t = ' '.join(t.strip().split())
		return t

	def remove_stopwords(self, corpus, remove_accents=False):
		stopwords = self.l_stopwords
		if remove_accents:
			stopwords = [self.strip_accents_nonalpha(w) for w in stopwords]
		pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
		return [pattern.sub('', text) for text in corpus]

	def remove_words(self, corpus, l_words):
		pattern = re.compile(r'\b(' + r'|'.join(l_words) + r')\b\s*')
		return [pattern.sub('', text) for text in corpus]

	def stemmer(self, corpus):
		stemmer = SnowballStemmer(self.lang)
		self.len_min_word = 1
		self.stem_map = {}

		def stem(word):
			stemmed = stemmer.stem(word)
			if stemmed not in self.stem_map:
				self.stem_map[stemmed] = []
			self.stem_map[stemmed].append(word)
			return stemmed

		corpus = [[stem(s) for s in x.split() if len(s) > self.len_min_word] for x in corpus]
		return [' '.join(x) for x in corpus]

	def re_stemmer(self, word):
		if not hasattr(self, 'stem_counters'):
			self.stem_counters = {w: Counter(self.stem_map[w]) for w in self.stem_map}
		return self.stem_counters[word].most_common(1)[0][0]

	def preprocess(self, corpus, stop_words=True, stemmer=True):
		self.l_stopwords = nltk.corpus.stopwords.words(self.lang)
		self.len_min_word = 1

		def do_preprocessing(l_docs):
			logging.info("preprocessing "+str(len(l_docs))+" documents")
			if stop_words:
				l_docs = self.remove_stopwords(l_docs, remove_accents=False)
			if stemmer:
				l_docs = self.stemmer(l_docs)
			l_docs = self.cleaning(l_docs)
			logging.info("done preprocessing")
			return l_docs

		if type(corpus) is dict:
			corpus = {k: do_preprocessing(corpus[k]) for k in corpus.keys()}
		else:
			corpus = do_preprocessing(corpus)

		return corpus

	def fit(self, X, y=None):
		return self

	def transform(self, X, *_):
		return self.preprocess(X, stop_words=self.stop_words, stemmer=self.stem)

class Loader:

	def __init__(self):
		pass

	# load supervised input
	def from_files(self, path, encod="ISO-8859-1"):
		dirs = glob.glob(os.path.join(path, "*", ""))
		class_names = []
		class_idx = []
		cid = -1
		corpus = []
		for _dir in dirs:
			cid += 1
			class_names.append(_dir.rstrip('\/').split('/')[-1])
			arqs = glob.glob(os.path.join(_dir, '*'))
			for arq in arqs:
				with codecs.open(arq, "r", encod) as myfile:
					data=myfile.read().replace('\n', '')
				corpus.append(data)
				class_idx.append(cid)
		result = {'corpus': corpus, 'class_index': class_idx, 'class_names': class_names}
		return result


	def from_files_2(self, path, encod="UTF-8"):
		corpus = []
		for arq in glob.iglob(path):
			with codecs.open(arq, "r", encod) as myfile:
				corpus.append(myfile.read().replace('\n',''))
		return corpus


	def from_text_line_by_line(self, arq):
		doc = []
		for line in open(arq):
			doc.append(line)
		return doc

	def _str_to_list(self, s):
		_s = re.split(',|{|}',s)
		return [ x for x in _s if len(x) > 0]

	def _str_to_date(self, s):
		pass

	def _convert(self, x, i, attr_list):
		if attr_list[i][1] == self.attr_numeric[1]:
			return float(x)
		elif attr_list[i][1] == self.attr_numeric[2]:
			return int(x)
		elif attr_list[i][1] == self.attr_string[0]:
			return x.replace("'","").replace('\'',"").replace('\"',"")
		else:
			return x.replace("'","").replace('\'',"").replace('\"',"")


	def from_arff(self, arq, delimiter=','):
		relation_name = ''
		attr_count = 0
		attr_list = []
		data = []
		self.attr_numeric = ['numeric', 'real', 'integer']
		self.attr_string = ['string']
		self.attr_date = ['date']
		read_data = False
		for line in open(arq):
			line = line.lower().strip()
			if line.startswith('#'): continue
			if read_data:
				vdata = line.split(delimiter)
				data.append([ self._convert(x,i,attr_list) for i,x in enumerate(vdata) ])
			elif not line.startswith('#'):
				if line.startswith('@relation'):
					relation_name = line.split()[1]
				elif line.startswith('@attribute'):
					attr_count += 1
					attr = line.split()
					attr_type = attr[2]
					if attr_type in self.attr_numeric or attr_type in self.attr_string:
						attr_list.append((attr[1], attr[2]))
					elif attr_type in self.attr_date:
						attr_list.append((attr[1], self._str_to_date(attr[2])))
					else:
						attr_list.append((attr[1], self._str_to_list(''.join(attr[2:]))))
				elif line.startswith('@data'):
					read_data = True
					continue
		d = dict()
		d['attributes'] = attr_list
		d['data'] = data
		d['relation'] = relation_name
		return d

	def from_sparse_arff(self,arq, delimiter=','):
		pass

class ConfigLabels:

	def __init__(self, unlabelled_idx=-1, list_n_labels=[10, 20, 30, 40, 50]):
		self.unlabelled_idx = unlabelled_idx
		self.list_n_labels = list_n_labels

	def pick_n_labelled(self, y, n_labelled_per_class):
		class_idx = set(y)
		labelled_idx = []
		for c in class_idx:
			r=numpy.isin(y, c)
			labelled_idx = numpy.concatenate((labelled_idx, numpy.random.choice(numpy.where(r)[0], n_labelled_per_class)))
		return labelled_idx.astype(int)

	# colocar o valor self.unlabelled_idx nos exemplos não rotulados de y
	def config_labels(self, y,labelled):
		unlabelled = []
		for i in range(len(y)):
			if i not in labelled:
				y[i] = self.unlabelled_idx
				unlabelled.append(i)
		return unlabelled

	# Return a dictionary key=<number of labels>, value is a list: [vector
	# with unlabels and labels, vector only with unlabels]
	def select_labelled_index(self, y, n_labels=[10, 20, 30, 40, 50]):
		dict_y = {}
		# Pick ni documentos rotulados por classe
		for ni in n_labels:
			dict_y[ni] = [numpy.array(y), None]
			nl = self.pick_n_labelled(y,ni)
			unl = self.config_labels(dict_y[ni][0],nl)
			dict_y[ni][1] = unl
		return dict_y

	def fit(self, y):
		dict_y = self.select_labelled_index(y, n_labels=self.list_n_labels)
		self.unlabelled_idx = {k:dict_y[k][1] for k in dict_y}
		self.semi_labels = { k:dict_y[k][0] for k in dict_y}
		return self

class RandMatrices:

	def create_rand_maps(self, D, W, K):
		A = self.create_rand_matrix_A(D, K)
		B = self.create_rand_matrix_B(W, K)
		Amap = dict()
		Bmap = dict()
		for j, d_j in enumerate(D):
			Amap[d_j] = A[j]
		for i, w_i in enumerate(W):
			Bmap[w_i] = B[i]
		return Amap, Bmap

	def create_rand_matrices(self, D, W, K):
		return self.create_rand_matrix_A(D, K), self.create_rand_matrix_B(W, K)

	def create_rand_matrix_B(self, W, K):
		N = len(W) # Number of words
		return numpy.random.dirichlet(numpy.ones(N), K).transpose() # B (N x K) matrix

	def create_rand_matrix_A(self, D, K):
		M = len(D) # Number of documents
		return numpy.random.dirichlet(numpy.ones(K), M)	# A (M x K) matrix

	def create_ones_matrix_A(self, D, K):
		M = len(D) # Number of documents
		return numpy.ones(shape=(M,K))

	def create_label_init_matrix_B(self, M, D, y, K, beta=0.0, unlabelled_idx=-1):
		ndocs, nwords = M.shape
		B = numpy.full((nwords, K),beta)
		count = {}
		for word in range(nwords):
			count[word] = defaultdict(int)
		rows, cols = M.nonzero()
		for row, col in zip(rows,cols):
			label = y[row]
			if label != unlabelled_idx:
				count[col][y[row]] += M[row,col]
				count[col][-1] += M[row,col]
		for word in range(nwords):
			for cls in count[word]:
				if cls != -1: B[word][cls] = (beta + count[word][cls])/(beta + count[word][-1])
		return B

	def create_label_init_matrices(self, X, D, W, K, y, beta=0.0, unlabelled_idx=-1):
		return self.create_rand_matrix_A(D, K), self.create_label_init_matrix_B(X, D, y, K, beta, unlabelled_idx)

	def create_fromB_matrix_A(self, X, D, B):
		K = len(B[0])
		M = len(D)	# number of documents
		A = numpy.zeros(shape=(M,K))
		for d_j in D:
			for w_i, f_ji in zip(X.indices[X.indptr[d_j]:X.indptr[d_j+1]], X.data[X.indptr[d_j]:X.indptr[d_j+1]]):
				A[d_j] += f_ji * B[w_i]
		return A

	def create_fromA_matrix_B(self, A):
		K = len(A[0])
		N = self.G.b_len()	 # number of words
		B = numpy.zeros(shape=(N,K))
		for w_i in self.G.b_vertices():
				for d_j, f_ji in self.G.w_b_neig(w_i):
						B[w_i] += f_ji * A[d_j]
		return self.normalizebycolumn(B)
