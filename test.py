import numpy as np
import functionspr1a as fnc
import pdb

'''
H=np.identity(3)
I=np.identity(5)


A=fnc.convolfn(I,H)
#print(np.array(H[0:3,0:3]))

print(A)

'''
'''
a = [[1, 2], [3, 4]]
#a_1=np.pad(a, ((3, 3), (3, 3)), 'minimum')
a_1=np.pad(a, ((1, 2), (1, 3)), 'constant')
#pdb.set_trace( )
print(a_1)
'''
'''
a = np.array([[1, 2], [3, 4]])
b = 2*a
c=a*b
print(c)
'''
A=[1,2,3,4,5,6,7,8,9,10,16,12,15]
for i in range(3,10):
	print(A[i-3])