import numpy as np
import sys
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils
from keras.models import load_model

#load data

data = np.load(sys.argv[1])

#This part create all model but use a lot of time
'''
loss = np.array([])
for i in range(200):
    print(i)
    x = data[str(i)]
    model = Sequential()

    model.add(Dense(units=100, activation='linear', input_shape = (100,)))

    model.add(Dense(units=80, activation='elu'))

    model.add(Dense(units=60, activation='elu'))

    model.add(Dense(units=80, activation='elu'))

    model.add(Dense(units=100, activation='elu'))

    model.add(Dense(units=100, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    
    model.fit(x, x, batch_size=100, epochs=20)


    loss = np.append(loss,[model.evaluate(x, x)])
    model.save('model%d.h5' %(i))
'''


#This part load all model and compute result but also use a lot of time

'''
loss = np.array([])
for i in range(200):
    print(i)
    x = data[str(i)]
    lmodel = load_model('model%d.h5' %(i))
    loss = np.append(loss,[lmodel.evaluate(x, x)])
'''

#This part read loss 

loss = genfromtxt('loss.csv',delimiter=',')

#This part  translate loss to dim.

ansdim = np.zeros(200)
addrate = 60./200.
temploss = np.zeros(200)

for i in range(200):
    temploss[i] = loss[i]


for j in range(200):
    for i in range(200):
        if temploss[i] == np.min(temploss):
            ansdim[i] = (j+1) * addrate
            temploss[i] = 500
            break


#This part write answer

newfile = open(sys.argv[2],'w')
newfile.write('SetId,LogDim\n')
setid = 0
for i in range(200):
    newfile.write("%d,%f\n" %(setid,np.log(ansdim[i])))
    setid = setid+1
newfile.close()
