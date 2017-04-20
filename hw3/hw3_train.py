import numpy as np
import sys
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils
from keras.models import load_model
#f = open(sys.argv[1],'r')
#newf = open('newtrain.csv','w')
#print('create newtrain.csv...')
#for line in f:
#	if len(line)>100:
#		newf.write(line.replace(',',' '))
#newf.close()

print('newtrain.csv created...')
print('get data...')
trainD = genfromtxt('newtrain.csv',delimiter=' ')
print('get data suc...')
print('divide data...')
x_train = trainD[:,1:]
X_train = x_train/255.
y_train = trainD[:,0]
print('divide data suc...')
print('xyshape')
print(x_train.shape)
print(y_train.shape)
y_train = np_utils.to_categorical(y_train,7)


#######
print('two clayer 10 20 one softmax')
model = Sequential()

model.add(Conv2D(64,(3,3), input_shape=(48,48,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3)))
model.add(MaxPooling2D((2,2)))


#model.add(Conv2D(30,(4,4)))
#model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=7,activation='softmax'))

x_train = x_train.reshape(x_train.shape[0],48,48,1)
opt = Adam()
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=100,epochs=20)
model.save('model0.h5')
score=model.evaluate(x_train,y_train)

f = open(sys.argv[2],'r')
newf = open('newtest.csv','w')
print('create newtest.csv...')
for line in f:
	if len(line)>100:
		newf.write(line.replace(',',' '))
newf.close()

inputD = genfromtxt('newtest.csv',delimiter=' ')
x_test = inputD[:,1:]
x_test = x_test.reshape(x_test.shape[0],48,48,1)

model = load_model('model0.h5')
result=model.predict(x_test)
result = np.argmax(result,axis=1)
out = open(sys.argv[3],'w')
out.write('id,label\n')
i=0
for x in result:
	out.write("%d,%d\n" %(i,x))
	i=i+1
#print(score[1])

'''
##########
print('two clayer 10 20 one dence 7 softmax')
model2 = Sequential()

model2.add(Conv2D(3,(3,3), input_shape=(48,48,1)))
model2.add(MaxPooling2D((2,2)))

model2.add(Conv2D(10,(3,3)))
model2.add(MaxPooling2D((2,2)))

model2.add(Conv2D(20,(3,3)))
model2.add(MaxPooling2D((2,2)))



model2.add(Flatten())
#model.add(Dense(units=7,activation='relu'))
model2.add(Dense(units=7,activation='softmax'))

model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model2.fit(x_train,y_train,batch_size=100,epochs=5)
score=model2.evaluate(x_train,y_train)
print(score[1])
'''
