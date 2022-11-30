import numpy as np
import pickle as pkl

corpus = np.load("newdata/corpus.npy", allow_pickle=True)

print("-".join([str(i) for i in corpus[0]]))

ogcorpus = np.load("data/corpus.npy")
print("-".join([str(i) for i in ogcorpus[0]]))

gen = np.load("data/gen_corpus.npy")
train = np.load("data/train_corpus.npy")
test = np.load("data/test_corpus.npy")
eval = np.load("data/eval_corpus.npy")


# print(corpus.shape, gen.shape, train.shape, test.shape, eval.shape)
print(gen)
# print(train)
# print(test)
# print(eval)
print(corpus)
