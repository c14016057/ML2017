import numpy as np
from numpy import genfromtxt
from numpy.linalg import inv
import matplotlib.pyplot as plt
import sys
trainD = genfromtxt(sys.argv[1], delimiter=',')
outcomeD = genfromtxt(sys.argv[2],delimiter=',')
lowScale = genfromtxt('lowScale.csv',delimiter=',')
lowup93 = lowScale > 0.9
lowdown60 = lowScale <0.6
#print(lowup93)
trainD = trainD[1:,:]
origD = trainD
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

train2D = np.array([trainD[:,0]]).T
train2D = np.hstack((train2D,trainD[:,2:6]))
#train2D = np.vstack((train2D, trainD[:,5])).T
#trainD = trainD[:,:6]
trainD = np.hstack((train2D, tempD))
trainD = np.hstack((trainD, temp2D))
trainD = np.hstack((trainD, temp3D))
trainD = np.hstack((trainD, temp4D))
trainD = np.hstack((trainD, temp5D))
trainD = np.hstack((trainD, temp6D))
trainD = np.hstack((trainD, temp7D))
#print(trainD)
#print(lowScale)
		###Scaling###
#meanD = np.mean(trainD,axis=0)
#stdD = np.std(trainD,axis=0)
#trainD = (trainD - meanD)/stdD
#trainD[:,0] = (np.ones((trainD.shape[0],1)))[:,0]

		###draw###
# y = hours_per_week
# x = age
# >50k = green
# <+50k = red
"""
for i in range(1000):
	if outcomeD[i] == 0 :
		plt.plot(trainD[i,0], trainD[i,1], 'ro')
	else :
		plt.plot(trainD[i,0], trainD[i,1], 'g^')
plt.show()
"""


"""
		###anylias###
hight = 0.
hightv = 0.
low = 0.
lowv = 0.
for i in range(trainD.shape[0]):
	if outcomeD[i] == 1:
		hight +=1
		hightv += trainD[i,1]
	else:
		low +=1
		lowv += trainD[i,1]
print('hight')
print(hight)
print(hightv/hight)
print('low')
print(low)
print(lowv/low)


"""

		###divide to two class###
lowD = np.zeros((1,(trainD.shape)[1]))
hightD = np.zeros((1,(trainD.shape)[1]))
for i in range((trainD.shape)[0]):
	if outcomeD[i] == 0:
		lowD = np.vstack((lowD,trainD[i]))
	else:
		hightD = np.vstack((hightD,trainD[i]))
lowD = lowD[1:,:]
hightD = hightD[1:,:]
print("devide success...")

		###compute mu###
lowMu = np.array([np.mean(lowD, axis = 0)])
hightMu = np.array([np.mean(hightD, axis = 0)])
print("compute Mu success...")

#lowSum = np.sum(lowD, axis = 0)
#hightSum = np.sum(hightD, axis = 0)
#print('lowSum')
#a = lowSum[2]
#print(a)
#a = lowMu[0,:]*lowD.shape[0]
#b = hightMu[0,:]*hightD.shape[0]
#writef = open("lowScale.csv","w")
#for i in range(lowScale.shape[0]):
#	writef.write("%f\n" %(lowScale[i]))
#b = hightSum[2]
#print('hightSum')
#print(b)
#print('lowScale')
#print(a/(a+b))



		###compute sigma###
lowSig = np.zeros((trainD.shape[1],trainD.shape[1]))
for i in range(lowD.shape[0]):
	lowx = np.array([lowD[i]])
	lowSig += np.dot( (lowx - lowMu).T, (lowx - lowMu))

hightSig = np.zeros((trainD.shape[1],trainD.shape[1]))
for i in range(hightD.shape[0]):
	hightx = np.array([hightD[i]])
	hightSig += np.dot( (hightx - hightMu).T,(hightx - hightMu))

avgSig = (hightSig + lowSig)/trainD.shape[0]
#print('avgSig')
#print(avgSig)
print("compute Sig success...")


		###compute gaussian function###
offsetG = ((2*np.pi)**(trainD.shape[1]/2.0)) * (np.linalg.det(avgSig)**0.5)
#print('offsetG')
#print(offsetG)
lowP = lowD.shape[0] / trainD.shape[0]
hightP = hightD.shape[0] / trainD.shape[0]
accept = 0
error = 0
out = open(sys.argv[4],"w")
out.write("id,label\n")


###Read test###
trainD = genfromtxt(sys.argv[3], delimiter=',')
trainD = trainD[1:,:]
###company###
tempD = np.array(trainD[:,6:15])
tempD *=lowScale[6:15]
tempD = np.sum(tempD, axis = 1)
tempD = (np.array([tempD])).T
print(tempD)

###school###
temp2D = np.array(trainD[:,15:31])
temp2D *=lowScale[15:31]
temp2D = np.sum(temp2D, axis = 1)
temp2D = (np.array([temp2D])).T
print(temp2D)

###marry status###
temp3D = np.array(trainD[:,31:38])
temp3D *=lowScale[31:38]
temp3D = np.sum(temp3D, axis = 1)
temp3D = (np.array([temp3D])).T
print(temp3D)

###job###
temp4D = np.array(trainD[:,38:53])
temp4D *=lowScale[38:53]
temp4D = np.sum(temp4D, axis = 1)
temp4D = (np.array([temp4D])).T
print(temp4D)

###job###
temp5D = np.array(trainD[:,53:59])
temp5D *=lowScale[53:59]
temp5D = np.sum(temp5D, axis = 1)
temp5D = (np.array([temp5D])).T
print(temp5D)

###color###
temp6D = np.array(trainD[:,59:64])
temp6D *=lowScale[59:64]
temp6D = np.sum(temp6D, axis = 1)
temp6D = (np.array([temp6D])).T
print(temp6D)

###native###
temp7D = np.array(trainD[:,64:106])
temp7D *=lowScale[64:106]
temp7D = np.sum(temp7D, axis = 1)
temp7D = (np.array([temp7D])).T
print(temp7D)

train2D = np.array([trainD[:,0]]).T
train2D = np.hstack((train2D,trainD[:,2:6]))
#train2D = np.vstack((train2D, trainD[:,5])).T
#trainD = trainD[:,:6]
trainD = np.hstack((train2D, tempD))
trainD = np.hstack((trainD, temp2D))
trainD = np.hstack((trainD, temp3D))
trainD = np.hstack((trainD, temp4D))
trainD = np.hstack((trainD, temp5D))
trainD = np.hstack((trainD, temp6D))
trainD = np.hstack((trainD, temp7D))
		###Scaling###
#meanD = np.mean(trainD,axis=0)
#stdD = np.std(trainD,axis=0)
#trainD = (trainD - meanD)/stdD
#trainD[:,0] = (np.ones((trainD.shape[0],1)))[:,0]

target = trainD
for i in range(target.shape[0]):
	x = np.array([target[i,:]])
	lowG = np.exp( np.dot( np.dot( (x-lowMu) , inv(avgSig) ) , (x-lowMu).T )* (-0.5)) / offsetG
	hightG = np.exp( np.dot( np.dot( (x-hightMu) , inv(avgSig) ) , (x-hightMu).T) * (-0.5)) / offsetG
	testlow = lowG*lowP / (lowG*lowP + hightG*hightP)
#	print('%f %d'%(testlow, outcomeD[i]))
#	avgpab += testlow
	predict = 0
	count = 0
	if testlow < 0.5:
		predict = 1
	out.write("%d,%d\n"%(i+1,predict))
	"""
	else:
		if testlow > 0.65:
			predict = 0
		else :
			count = 0
			for j in range (lowup93.shape[0]):
				if lowup93[j] == True and origD[i,j] == 1 :
					count +=1
			if count > 1 :
				predict = 0
				print('0:%f %d'%(testlow, outcomeD[i]))
			else :
				predict = 1
				print('1:%f %d'%(testlow, outcomeD[i]))
	
	for j in range (lowup93.shape[0]):
		if lowup93[j] == True and origD[i,j] == 1 :
			count +=1
	if count > 1 :
		predict = 0
		#print('0:%f %d'%(testlow, outcomeD[i]))
	count = 0
	for j in range (lowdown60.shape[0]):
		if lowdown60[j] == True and origD[i,j] == 1 :
			count +=1
	print('%d %f %d'%(count,testlow, outcomeD[i]))
	"""

##		print('smaller')
#	if predict == outcomeD[i]:
##		print('accept')
#		accept +=1
	#else:
##		print('error')
	#	error +=1
#print("accruy:")
#print(accept/target.shape[0])
#print("avgpab")
#print(avgpab/target.shape[0])


