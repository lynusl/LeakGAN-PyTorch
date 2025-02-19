from main import restore_checkpoint, generate_samples
import numpy as np
from functools import reduce
import pickle as pkl

def tensor_to_text(input_x, vocab):
    #vocab will convert some integer into a word
    poem = []
    sen_len = 50
    for index, x in enumerate(input_x):
        if index != 0:
            if index % sen_len == 0:
                poem.append("\n")
        poem.append(vocab[x-1]) #x-1 because when encoding we added one
    poem_ = ""
    poem_ = reduce((lambda x, y:x + y), poem) # just add strings
    print(poem_) #to see
    return poem_

# checkpoint = restore_checkpoint("checkpoint0.pth.tar")

# model_dict = checkpoint["model_dict"]
# optimizer_dict = checkpoint["optimizer_dict"]
# scheduler_dict = checkpoint["scheduler_dict"]
# ckpt_num = checkpoint["ckpt_num"]

# generate_samples(model_dict, "newdata/finalgenerated_0", 16, use_cuda=True)

# final = np.load("newdata/finalgenerated_0.npy")
# print(final)

# print(final.shape)

vocablist = list("VINGTCYLE AMWPHRQSFDK")

f = open("newdata/chars.pkl", "wb") #this is in a sense table of keys
pkl.dump(vocablist, f)
f.close()

# corpus = np.load("newdata/corpus.npy", allow_pickle=True)

# for j in range(18,19):
#     print("-".join([str(i) for i in corpus[j]]))

###################

# print(final[0].dtype)

# with open("newdata/chars.pkl", 'rb') as vocabf:
#     vocab = pkl.load(vocabf)

# print(vocab)

# tensor_to_text(final[0], vocablist)

# with open("newdata/generated_sequences_0.txt", 'w') as genseqtxt:
#     genseqtxt.writelines([tensor_to_text(line, vocablist).strip()+"\n" for line in final])