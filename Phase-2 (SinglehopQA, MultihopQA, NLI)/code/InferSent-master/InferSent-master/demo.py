# import stuff


from random import randint

import numpy as np
import torch



# Load model
from models import InferSent
model_version = 1
MODEL_PATH = "D:/anc final project/InferSent-master/InferSent-master/encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))



# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model


# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'D:/anc final project/InferSent-master/InferSent-master/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)


# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)



# Load some sentences
sentences = []
with open('samples.txt') as f:
    for line in f:
        sentences.append(line.strip())
print(len(sentences))



sentences[:5]


embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings)))


np.linalg.norm(model.encode(['the cat eats.']))




def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))





cosine(model.encode(['the cat eats.'])[0], model.encode(['the cat drinks.'])[0])






idx = randint(0, len(sentences))
_, _ = model.visualize(sentences[idx])




my_sent = 'Jack is going to Japan tonight. Mary just returned from Nairobi.'
_, _ = model.visualize(my_sent)




model.build_vocab_k_words(500000) # getting 500K words vocab
my_sent = 'Cassandra and Epictetus are having a fight.'
_, _ = model.visualize(my_sent)
























