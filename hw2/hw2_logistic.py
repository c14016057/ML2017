import numpy as np
from numpy import genfromtxt
from numpy.linalg import inv
import matplotlib.pyplot as plt
import sys
trainD = genfromtxt(sys.argv[1], delimiter=',')
outcomeD = np.array( [genfromtxt(sys.argv[2],delimiter=',')]).T
lowScale = genfromtxt('lowScale.csv',delimiter=',')
print(trainD.shape)
print(outcomeD.shape)
trainD = trainD[1:,:]

###company###
tempD = np.array(trainD[:,6:15])
tempD *=lowScale[6:15]
tempD = np.sum(tempD, axis = 1)
tempD = (np.array([tempD])).T

###school###
temp2D = np.array(trainD[:,15:31])
temp2D *=lowScale[15:31]
temp2D = np.sum(temp2D, axis = 1)
temp2D = (np.array([temp2D])).T

###marry status###
temp3D = np.array(trainD[:,31:38])
temp3D *=lowScale[31:38]
temp3D = np.sum(temp3D, axis = 1)
temp3D = (np.array([temp3D])).T

###job###
temp4D = np.array(trainD[:,38:53])
temp4D *=lowScale[38:53]
temp4D = np.sum(temp4D, axis = 1)
temp4D = (np.array([temp4D])).T

###job###
temp5D = np.array(trainD[:,53:59])
temp5D *=lowScale[53:59]
temp5D = np.sum(temp5D, axis = 1)
temp5D = (np.array([temp5D])).T

###color###
temp6D = np.array(trainD[:,59:64])
temp6D *=lowScale[59:64]
temp6D = np.sum(temp6D, axis = 1)
temp6D = (np.array([temp6D])).T

###native###
temp7D = np.array(trainD[:,64:106])
temp7D *=lowScale[64:106]
temp7D = np.sum(temp7D, axis = 1)
temp7D = (np.array([temp7D])).T

#train2D = np.array([trainD[:,0]]).T
#train2D = np.hstack((train2D,trainD[:,2:6]))
#train2D = np.vstack((train2D, trainD[:,5])).T
train2D = trainD[:,:6]
trainD = np.hstack((train2D, tempD))
trainD = np.hstack((trainD, temp2D))
trainD = np.hstack((trainD, temp3D))
trainD = np.hstack((trainD, temp4D))
trainD = np.hstack((trainD, temp5D))
trainD = np.hstack((trainD, temp6D))
trainD = np.hstack((trainD, temp7D))

#for i in range(300):
#	if outcomeD[i] == 0:
#		plt.plot(trainD[i,7],trainD[i,6], 'ro')
#	else:
#		plt.plot(trainD[i,7],trainD[i,6],'g^')
#plt.show()
trainDL = np.hstack((trainD, outcomeD))
trainDL = np.hstack( (np.ones((trainDL.shape[0],1)), trainDL))



		### fs###
X = trainDL[:,:-1]
y = trainDL[:,trainDL.shape[1]-1]
ansW = (np.dot( np.dot( inv(np.dot(X.T, X)) , X.T) , y)).T 
print(ansW.shape)
startW = np.hstack((ansW,np.ones(1)*(-1)))
wdSumRow = (startW*trainDL).sum(axis=1)
print((wdSumRow*wdSumRow).sum())
"""

		### CREATE PARTIONAL DIR. ###
pd = trainDL[:,:-1]
pd = np.transpose(pd)

		### Init. some para.###
startW = np.zeros((1,trainDL.shape[1]))
startW[0,trainDL.shape[1]-1] = -1
dw = np.zeros((trainDL.shape[1]-1,1))
dwsum = np.zeros((trainDL.shape[1]-1,1))

		### TRANIING ###
for i in range(3000):
	print("i")
	print(i)
	if i%50 == 0:
		dwsum /= 2
	ndw = np.zeros((1,trainDL.shape[1]))
	ndw[0,:-1] = dw.T
	startW -=ndw
#	print("startw")
#	print(startW)
	wd = startW*trainDL

	### Train error ###
	print("train error")
	wdSumRow = wd.sum(axis=1)
	print((wdSumRow*wdSumRow).sum())

#print(twowd)
#print("[pd]*[2WD]")
	befsum = np.dot(pd,wd)
#print(befsum)





	sumrow = np.ones((trainDL.shape[1],1))
	dw = np.dot(befsum,sumrow)
#	print("orig dw")
#	print(dw)
###sigu###
#	landa = 100
#	WW = startW[:,:-1].T
#	WW2 = WW**2
#	dw += landa*WW2.sum()
#	dw = dw - landa*WW2 
#	dw = dw + 2*landa*WW
###adma###
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

###Read test###
trainD = genfromtxt(sys.argv[3], delimiter=',')
trainD = trainD[1:,:]

###company###
tempD = np.array(trainD[:,6:15])
tempD *=lowScale[6:15]
tempD = np.sum(tempD, axis = 1)
tempD = (np.array([tempD])).T
#print(tempD)

###school###
temp2D = np.array(trainD[:,15:31])
temp2D *=lowScale[15:31]
temp2D = np.sum(temp2D, axis = 1)
temp2D = (np.array([temp2D])).T
#print(temp2D)

###marry status###
temp3D = np.array(trainD[:,31:38])
temp3D *=lowScale[31:38]
temp3D = np.sum(temp3D, axis = 1)
temp3D = (np.array([temp3D])).T
#print(temp3D)

###job###
temp4D = np.array(trainD[:,38:53])
temp4D *=lowScale[38:53]
temp4D = np.sum(temp4D, axis = 1)
temp4D = (np.array([temp4D])).T
#print(temp4D)

###job###
temp5D = np.array(trainD[:,53:59])
temp5D *=lowScale[53:59]
temp5D = np.sum(temp5D, axis = 1)
temp5D = (np.array([temp5D])).T
#print(temp5D)

###color###
temp6D = np.array(trainD[:,59:64])
temp6D *=lowScale[59:64]
temp6D = np.sum(temp6D, axis = 1)
temp6D = (np.array([temp6D])).T
#print(temp6D)

###native###
temp7D = np.array(trainD[:,64:106])
temp7D *=lowScale[64:106]
temp7D = np.sum(temp7D, axis = 1)
temp7D = (np.array([temp7D])).T
#print(temp7D)

#train2D = np.array([trainD[:,0]]).T
#train2D = np.hstack((train2D,trainD[:,2:6]))
train2D = trainD[:,:6]
trainD = np.hstack((train2D, tempD))
trainD = np.hstack((trainD, temp2D))
trainD = np.hstack((trainD, temp3D))
trainD = np.hstack((trainD, temp4D))
trainD = np.hstack((trainD, temp5D))
trainD = np.hstack((trainD, temp6D))
trainD = np.hstack((trainD, temp7D))
#print(trainD)

out = open(sys.argv[4],"w")
out.write("id,label\n")

target = trainD
for r in range(1):
	accept = 0.
	for i in range(target.shape[0]):
		x = np.array([target[i,:]])	

		x = np.hstack( (np.ones((1,1)),x))
		testhight = (x*ansW).sum()
		logis = testhight
		if testhight > (0.46):
			testhight = 1
		else:
			testhight = 0
		#if(testhight!=predict):
			#print('log:%f,lowp%f,ans:%d'%(logis,testlow,outcomeD[i]))
		
		out.write("%d,%d\n"%(i+1,testhight))
		#print('higp:%f ans:%d'%(1+logis-testlow, outcomeD[i]))
	#	if testhight == outcomeD[i]:
	#		accept += 1.
		#else:
#	print('accury')
#	print(accept/target.shape[0])

