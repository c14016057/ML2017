import word2vec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from adjustText import adjust_text
import nltk
#nltk.download()

word2vec.word2phrase('all.txt', 'all-phrases', verbose=True)

word2vec.word2vec('all-phrases', 'all.bin', size=200, verbose=False)

model = word2vec.load('all.bin')

#get vector and label
wordnum = 500
fword = model.vocab[:wordnum]
#print(fword)
x = model.vectors[:wordnum,:]

#tsne
tmodel = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
wvector = tmodel.fit_transform(x)
#print(wvector)

#draw
s_tag(fword)
texts = []
plt.figure(figsize=(15, 15))
for i in range(wordnum):
    if(wtag[i][1] == 'JJ' or 
       wtag[i][1] == 'NN' or 
       wtag[i][1] == 'NNS' or
       wtag[i][1] == 'NNP'):
        texts.append(plt.text(wvector[i][0], wvector[i][1], wtag[i][0]))
        plt.plot(wvector[i][0],wvector[i][1],'o')
plt.title(str(adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))))
plt.show()

