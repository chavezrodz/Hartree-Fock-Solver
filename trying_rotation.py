import numpy as np
import matplotlib.pyplot as py

N_shape = (50,50)

#Allowed Momentum Indices for itterator
Qx,Qy = np.indices(N_shape,sparse=True)
Qx,Qy = Qx.flatten(),Qy.flatten()

# Allowed Momentum Values
Qxv = Qx*np.pi/N_shape[0] - np.pi/2
Qyv = Qy*np.pi/N_shape[1] - np.pi/2

angle = -np.pi / 4.
rotate = np.array(( (np.cos(angle), -np.sin(angle)),
               (np.sin(angle),  np.cos(angle)) ))
scale = np.array( ( (1./np.sqrt(2),0), (0,1./np.sqrt(2)) ) )
# e1 = rot.unitz # unit matrix, out of which we
#                                     # pick the rotation axis
# e2 = rot.unity # unit matrix, out of which we
#                                     # pick the rotation axis
# a1 = np.hstack((e1, (angle,))) # 4-tuple with the angle at the end
# R1 = rot.matrix_from_axis_angle(a1) # rotation matrix based on this
# a2 = np.hstack((e2, (angle,))) # 4-tuple with the angle at the end
# R2 = rot.matrix_from_axis_angle(a2) # rotation matrix based on this
# # now create the array, everything is multiplied by this rotation
Qv =np.array( [ np.dot(scale,np.dot(rotate,np.array([k_x,k_y]))) \
                for k_x in Qxv for k_y in Qyv ] )
# similarly create an array that houses the indices, in same order
Q = np.array( [np.array(ind_x,ind_y) for ind_x in Qx \
												for ind_y in Qy]  )

# now let us plot just to see what the BZ looks like
x_comp, y_comp = Qv.transpose()
py.scatter(x_comp, y_comp)
py.show()
