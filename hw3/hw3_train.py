import numpy as np
import sys
import pandas as pd
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

dim = 48*48
def loadData():
    traw = pd.read_csv(sys.argv[1], sep=',', delimiter=None, header=0)
    trawPA = np.array(traw)
    trows = trawPA.shape[0]
    tfeatures = np.empty([trows, dim])
    ids = np.array([])
    
    for i in range(trows):
        row = trawPA[i, :]
        ids = np.append(ids, row[0])
        feat = row[1].split()
        feat = list(map(int, feat))
        tfeatures[i, :] = feat
    tfeatures.astype('float32')
    tfeatures = tfeatures/255
    return [ids, tfeatures]
[id_test, x_test] = loadData()
#y_early = y_train[20000:,:]
#x_early = x_train[20000:,:]
#y_train = y_train[:20000,:]
#x_train = x_train[:20000,:]
#x_early = x_early.reshape(x_early.shape[0],48,48,1)
#x_train = x_train.reshape(x_train.shape[0],48,48,1)


model = load_model('model.h5')

x_test = x_test.reshape(x_test.shape[0],48,48,1)
result=model.predict(x_test)
result = np.argmax(result,axis=1)
out = open(sys.argv[2],'w')
out.write('id,label\n')
i=0
for x in result:
    out.write("%d,%d\n" %(i,x))
    i=i+1

