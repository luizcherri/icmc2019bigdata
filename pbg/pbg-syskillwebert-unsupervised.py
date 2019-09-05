#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pbg import PBG
from util import Loader
from util import Preprocessor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def get_top_words(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return topics

l = Loader()
d = l.from_files('/home/wise/Software/pbg/input/parsed/syskillwebert')
cvect = CountVectorizer()
steps = [('preprocessor', Preprocessor()), ('countvectorizer', cvect)]
pipe = Pipeline(steps)
pipe.fit(d['corpus'])
M = pipe.transform(d['corpus'])

number_of_topics = 50
words_by_topics = 10
model = None

model = PBG(n_components=number_of_topics, rand_init=True)
model.fit(M)
model.transform(M)
topics = get_top_words(model, cvect.get_feature_names(), words_by_topics)

print('Algorithm:', 'PBG')
print('Number of topics:', number_of_topics)
print('Words by topics:', words_by_topics)
for key, topic in enumerate(topics):
    print(key + 1, '[' + ', '.join(topic) + ']')
print('\n')

exit()
