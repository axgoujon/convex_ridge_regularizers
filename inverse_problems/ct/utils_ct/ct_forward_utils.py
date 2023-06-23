import odl
import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(__file__))

import torch_wrapper

############## specify geometry parameters #################
img_size, space_range = 512, 128 #space discretization
num_angles, det_shape = 200, 400 #projection parameters



#geom => 'parallel_beam'

###### build operator #############
# forward, adjoint and fbp operators
def get_operators(space_range=space_range, num_angles=num_angles, det_shape=det_shape, device=None, fix_scaling=True):

    space = odl.uniform_discr([-space_range, -space_range], [space_range, space_range],\
                              (img_size, img_size), dtype='float32', weighting=None)
    geometry = odl.tomo.geometry.parallel.parallel_beam_geometry(space, num_angles=num_angles, det_shape=det_shape)
   
    fwd_op_odl = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    fbp_op_odl = odl.tomo.fbp_op(fwd_op_odl)

    adjoint_op_odl = fwd_op_odl.adjoint

    # !!!! the ray transform for odl and astra_cuda is known to have scaling issues (related to the volume of the discretization cells)
    # this is a hack to fix it
    if fix_scaling:
        # fbp
        # we check the scaling on a circular mask
        fbp_op_odl = fbp_op_odl * get_fbp_scaling(fwd_op_odl, fbp_op_odl)
    
        # adjoint
        # we check the scaling by comparing <y, Hx>  and <H^T y, x> for a random input
        #adjoint_op_odl = adjoint_op_odl * get_adjoint_scaling(fwd_op_odl, adjoint_op_odl)
        scaling = get_adjoint_scaling(fwd_op_odl, adjoint_op_odl)
        print(scaling)
        #scaling = get_adjoint_scaling(fwd_op_odl, adjoint_op_odl)
        scaled_fwd_op_odl = fwd_op_odl * scaling
        adjoint_op_odl = scaled_fwd_op_odl.adjoint

    # torch wrapper
    fwd_op = torch_wrapper.OperatorModule(fwd_op_odl).to(device)
    fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device)
    adjoint_op = torch_wrapper.OperatorModule(adjoint_op_odl).to(device)
    
    return fwd_op, fbp_op, adjoint_op

###### build operator #############
# For some reason the adjoint has a scaling problem
# This is a hack to fix it
def get_adjoint_scaling(fwp, adjoint, inverse=True):
    x0 = np.random.rand(img_size,img_size)
    y0 = np.random.rand(num_angles, det_shape)


    Hx0 = fwp(x0)
    s1 = np.sum(y0*Hx0)

    s2 = np.sum(adjoint(y0)*x0)

    print(s1/s2, s2/s1)
    # return(s2/s1)
    return(s1/s2)

def get_fbp_scaling(fwd, fbp):

    x0 = create_circular_mask(img_size//2)

    sinogram = fwd(x0)
    y = fbp(sinogram)

    # check in the center
    y_center = y * x0

    return(x0.mean()/np.mean(y_center).item())
   
def create_circular_mask(radius):
    center = (int(img_size/2), int(img_size/2))
    

    Y, X = np.ogrid[:img_size, :img_size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius

    return mask


def get_op_norm(fwd_op, adjoint_op, device, n_iter=15):
    x = torch.rand((1, 1, img_size,img_size)).to(device).requires_grad_(False)

    with torch.no_grad():
        for i in range(n_iter):
            x = x / x.norm()
            x = adjoint_op(fwd_op(x))

    return(x.norm().sqrt().item())



if __name__ == '__main__':
    fwd_op, fbp_op, adjoint_op = get_operators()
    #get_norm_HtH(fwd_op, adjoint_op, device = 'cuda')

    
    
    
    
    
    
    


