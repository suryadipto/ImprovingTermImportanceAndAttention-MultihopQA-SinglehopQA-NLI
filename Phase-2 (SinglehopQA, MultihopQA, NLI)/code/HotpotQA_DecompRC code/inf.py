import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

train_raw_df = fetch_20newsgroups(subset='train', categories=categories)
test_raw_df = fetch_20newsgroups(subset='test', categories=categories)

x_train, x_val, y_train, y_val = train_test_split(np.array(train_raw_df.data), train_raw_df.target, test_size=0.1)
x_test = np.array(test_raw_df.data)
y_test = test_raw_df.target

# x_train = [x_train[:200] for x in x_train]

print('Train:', len(x_train))
print('Val:', len(x_val))
print('Test:', len(x_test))

import sys, os
def add_aion(curr_path=None):
    aion_dir = 'D:/anc final project/Infersent_SentEmbedding/aion'
    sys.path.insert(0, aion_dir)
            
add_aion()

import nltk
nltk.download('punkt')

from aion.embeddings.infersent import InferSentEmbeddings

infer_sent_embs = InferSentEmbeddings(word_embeddings_dir='D:/anc final project/glove.840B.300d/', verbose=20)
infer_sent_embs.load_model(dest_dir='D:/anc final project/InferSent/encoder/')

infer_sent_embs.build_vocab(x_train, tokenize=True)

x_train_t = infer_sent_embs.encode(x_train, tokenize=True)
x_test_t = infer_sent_embs.encode(x_test, tokenize=True)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='newton-cg', max_iter=1000)
model.fit(x_train_t, y_train)

y_pred = model.predict(x_test_t)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print('Accuracy:%.2f%%' % (accuracy_score(y_test, y_pred)*100))
print('Classification Report:')
print(classification_report(y_test, y_pred))

