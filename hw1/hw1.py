import numpy as np
from numpy import genfromtxt
from numpy.linalg import inv
import matplotlib.pyplot as plt
import sys
inputD = genfromtxt('train.csv', delimiter=',')
inputD = inputD[1:,3:]
baseI = 0
baseJ = 0
arrinput = np.ones((18,1))
for month in range(12):#12
	for day in range(20):#20
		arrinput = np.hstack((arrinput,inputD[baseI:baseI+18,0:24]))
		baseI += 18
arrinput = arrinput[:,1:]
whereNan = np.argwhere(np.isnan(arrinput))
for i in range((whereNan.shape)[0]):
	arrinput[whereNan[i,0],whereNan[i,1]] = 0
trainD = np.ones((1,164))
vone = np.array([1])
for month in range(12):#12
	baseM = month *480
	for hour in range(471):#471
		baseJ = baseM + hour
		trainBlock = arrinput[0:18,baseJ:baseJ+9]
		train25 = np.array([arrinput[9,baseJ+9]])
		trainDrow = (np.reshape(trainBlock,(1,162)))[0]
		trainDrow = np.hstack((vone,trainDrow))
		trainDrow = np.hstack((trainDrow,train25))
		trainD = np.vstack((trainD,trainDrow))
trainD = trainD[1:,:]
startW = np.zeros((1,164))

		###Scaling###
meanD = np.mean(trainD,axis=0)
stdD = np.std(trainD,axis=0)
ScaleTrainD = (trainD - meanD)/stdD
ScaleTrainD[:,0] = (np.ones((5652,1)))[:,0]
#ScaleTrainD[:,163] = trainD[:,163]
#startW *= -0.001
startW[0,163] = -1
dw = np.zeros((163,1))
dwsum = np.zeros((163,1))



		### CREATE PARTIONAL DIR. ###
pd = ScaleTrainD[:,:-1]
pd = np.transpose(pd)



		### READ MOUDLE ###
s = genfromtxt('moudle_665.csv', delimiter=',')
scale = np.array([s])
#scale = np.abs(scale.T)

print(scale)
"""
		### foumla sol###
X = ScaleTrainD[:,:-1]
y = ScaleTrainD[:,163]
ansW = (np.dot( np.dot( inv(np.dot(X.T, X)) , X.T) , y)).T 
startW = np.hstack((ansW,vone*(-1)))
wdSumRow = (startW*ScaleTrainD).sum(axis=1)
print((wdSumRow*wdSumRow).sum())
"""
"""
		### TRANIING ###
for i in range(1000):
	print("i")
	print(i)
	if i%50 == 0:
		dwsum /= 2
	ndw = np.zeros((1,164))
	ndw[0,:-1] = dw.T
	startW -=ndw
#	print("startw")
#	print(startW)
	wd = startW*trainD
	wd_scal = startW*ScaleTrainD
	twowd = 2*wd_scal

	### Train error ###
	print("train error")
	wdSumRow = wd_scal.sum(axis=1)
	print((wdSumRow*wdSumRow).sum())

#print(twowd)
#print("[pd]*[2WD]")
	befsum = np.dot(pd,twowd)
#print(befsum)





	sumrow = np.ones((164,1))
	dw = np.dot(befsum,sumrow)
#	print("orig dw")
#	print(dw)
###sigu###
	landa = 100
	WW = startW[:,:-1].T
	WW2 = WW**2
	dw += landa*WW2.sum()
	dw = dw - landa*WW2 
	dw = dw + 2*landa*WW
###sigu###
	dwsum += dw*dw
#	print("dwsum")
#	print(dwsum)
#	dw *= scale
	dw /= np.sqrt(dwsum)
#	print("dw")
#	print(dw)

print("ansW")
ansW = (startW[:,:-1])[0]
print(ansW)
"""
ansW = scale[0]
	###Create moudle###
moudle = open("newmoudle.csv","w")
for i in range((ansW.shape)[0]):
	moudle.write("%f\n" %(ansW[i]))

	### predict ###
out = open("newout.csv",'w')
out.write("id,value\n")
testD = genfromtxt('test_X.csv', delimiter=',')
testD = testD[:,2:]
testNan = np.argwhere(np.isnan(testD))
for i in range((testNan.shape)[0]):
	testD[testNan[i,0],testNan[i,1]] = 0
for i in range(240):#240
	I = testD[i*18:i*18+18,0:9]
	reI = (np.reshape(I,(1,162)))[0]
	reIO = np.hstack((vone,reI))
	reIO[1:163] -=meanD[1:163]
	reIO[1:163] /=stdD[1:163]
#	print(reIO)
#	print(ansW)
	pm25 = (reIO*ansW).sum()
	
	pm25 = pm25*stdD[163]+meanD[163]
	out.write("id_%d,%f\n" %(i,pm25))
	#print((reIO*ansW).sum())

