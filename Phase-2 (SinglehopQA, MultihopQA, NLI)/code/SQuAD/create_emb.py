# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:17:39 2020

@author: ssark
"""

import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import pandas as pd
import json
from textblob import TextBlob
import nltk
from scipy import spatial
import torch
import spacy
#en_nlp = spacy.load('en_core_web_sm')

train = pd.read_json("data/train-v1.1.json")
valid = pd.read_json("data/dev-v1.1.json")
train.shape, valid.shape

train.head(3)

train.iloc[1,0]['paragraphs'][0]

# valid.iloc[1,0]['paragraphs'][0]

contexts = []
questions = []
answers_text = []
answers_start = []
for i in range(train.shape[0]):
    topic = train.iloc[i,0]['paragraphs']
    for sub_para in topic:
        for q_a in sub_para['qas']:
            questions.append(q_a['question'])
            answers_start.append(q_a['answers'][0]['answer_start'])
            answers_text.append(q_a['answers'][0]['text'])
            contexts.append(sub_para['context'])   
df = pd.DataFrame({"context":contexts, "question": questions, "answer_start": answers_start, "text": answers_text})

df.shape

df.to_csv("data/train.csv", index = None)

paras = list(df["context"].drop_duplicates().reset_index(drop= True))

len(paras)

blob = TextBlob(" ".join(paras))
sentences = [item.raw for item in blob.sentences]

len(sentences)

infersent = torch.load('InferSent/encoder/infersent2.pkl', map_location=lambda storage, loc: storage)
infersent.set_w2v_path("ID:/anc final project/crawl-300d-2M.vec/glove.840B.300d.vec")
infersent.build_vocab(sentences, tokenize=True)

dict_embeddings = {}
for i in range(len(sentences)):
    print(i)
    dict_embeddings[sentences[i]] = infersent.encode([sentences[i]], tokenize=True)

questions = list(df["question"])

len(questions)

for i in range(len(questions)):
    print(i)
    dict_embeddings[questions[i]] = infersent.encode([questions[i]], tokenize=True)


dict_embeddings['Architecturally, the school has a Catholic character.'][0]


d1 = {key:dict_embeddings[key] for i, key in enumerate(dict_embeddings) if i % 2 == 0}
d2 = {key:dict_embeddings[key] for i, key in enumerate(dict_embeddings) if i % 2 == 1}
d1
d2

with open('data/dict_embeddings1.pickle', 'wb') as handle:
    pickle.dump(d1, handle)
    
with open('data/dict_embeddings2.pickle', 'wb') as handle:
    pickle.dump(d2, handle)
    
del dict_embeddings





