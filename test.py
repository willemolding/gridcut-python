import gridcut
from numpy import *

source = array([8,0,0,0,0,0,0,0,0],dtype=float32)
sink = array([0,0,0,0,0,0,0,9,0],dtype=float32)
up = array([5,1,0,0,0,0,0,0,0])
down =   array([0,0,4,0,0,0,0,3,0])
left =   array([0,0,0,0,3,0,0,0,0])
right =  array([0,0,5,0,0,0,0,0,0])

pw = ones(9,dtype=float32)*8.

#print gridcut.solve_2D_4C(3,3,source,sink,pw,pw,pw,pw,n_threads=1,block_size=4).reshape((3,3))


print gridcut.solve_2D_4C_potts(3,3,8,source,sink,n_threads=1,block_size=1).reshape((3,3))
print gridcut.solve_2D_4C_potts(3,3,8,source,sink,n_threads=1,block_size=1).reshape((3,3))