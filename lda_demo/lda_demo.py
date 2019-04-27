# pip install lda
import numpy as np
import lda
import lda.datasets
import lda.utils
import os
_test_dir = os.path.dirname(__file__)
reuters_ldac_fn = os.path.join(_test_dir + '/data/reuters.ldac')
print(reuters_ldac_fn)
X = lda.utils.ldac2dtm(open(reuters_ldac_fn), offset=0)
reuters_vocab_fn = os.path.join(_test_dir + '/data/reuters.tokens')
with open(reuters_vocab_fn) as f:
    vocab = tuple(f.read().split())
reuters_titles_fn = os.path.join(_test_dir + '/data/reuters.titles')
with open(reuters_titles_fn) as f:
    titles = tuple(line.strip() for line in f.readlines())
print(X.shape)
print(X.sum())

model = lda.LDA(n_topics=4, n_iter=2, random_state=1, refresh=2)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
