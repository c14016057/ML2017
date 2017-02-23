import sys
a = open (sys.argv[1], "r")
b = open (sys.argv[2], "r")
#c = open ("matrixC.txt", "w")
ans = open("ans_one.txt", "w")
matrixA = []
for line in a:
	matrixA.append ([int (v) for v in line.split (",")])
row = len (matrixA)
mid = len (matrixA[0])
matrixB = []
for line in b:
	matrixB.append ([int (v) for v in line.split (",")])
col = len (matrixB[0])
matrixC = []
for i in range (0, row, 1):
	for j in range (0, col, 1):
		s = 0
		for k in range (0, mid, 1):
			s += matrixA[i][k] * matrixB[k][j]
#		c.write (str (s) )
		matrixC.append(s)
"""		if j == 4 :
			c.write ("\n")
		else :
			c.write (" ")"""
matrixC.sort ()
for i in range (0, len(matrixC), 1):
	ans.write (str (matrixC[i]))
	ans.write ("\n")
a.close()
b.close()
#c.close()
	
