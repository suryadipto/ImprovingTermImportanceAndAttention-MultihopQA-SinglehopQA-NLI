Context-based Question Answering
---------------------------------

Problem statement: Our goal is to train a model which is capable of responding to complex questions, or questions based on convoluted text.

Motivation: Effective generation of context-based word embeddings to enable models to answer more complex, nuanced questions from a given piece of text.

Intuition: The model must have a notion of “context”; and must be able to distinguish not only amongst words, but also: their placement in the sentence, part of speech of the word et cetera. For common sense question answering, this might also mean that the model has some knowledge about “the ways of the world”.

Models to be used: BERT definitely, Elmo maybe, other frameworks depending on applicability and performance.

Language: Python 2.7 or higher

References:
[1] Multi-hop question answering: https://arxiv.org/pdf/1809.09600.pdf
[2] Commonsense question answering: https://www.aclweb.org/anthology/N19-1421.pdf

Expected outcome: Given a grammatically correct set of informative English sentences, the model should be able to answer questions based on it. The rate of questions answered is going to be a measure of accuracy of the trained model.

