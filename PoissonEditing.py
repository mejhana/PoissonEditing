import cv2
import json
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

class PoissonEditing:
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

    def read_json(self):
        f = open("annotation.json")
        ann = json.load(f)
        points = []
        for x in ann:    
            for zm in ann[x]['regions']:
                att = zm['shape_attributes']
                points.append([att['cx'], att['cy']])
        f.close()
        return points

    def get_mask(self):
        points = self.read_json()
        temp = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)
        mask = cv2.fillPoly(temp, pts=np.int32([points]), color=(255, 0, 0))
        mask = (mask == 255).astype(int)
        return points,  mask
    
    def laplacian_matrix(self,n, m):   
        mat_D = scipy.sparse.lil_matrix((m, m))
        mat_D.setdiag(-1, -1)
        mat_D.setdiag(4)
        mat_D.setdiag(-1, 1)
            
        mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
        
        mat_A.setdiag(-1, 1*m)
        mat_A.setdiag(-1, -1*m)
    
        return mat_A
    
    def poisson_edit(self, source, mask, dest, cloning=False, grad_mix=False):
        if cloning:
            assert mask.shape == dest.shape, "Ensure that the source and target are of same shape"
            y_range, x_range = dest.shape    
            
        else:
            assert mask.shape == source.shape, "Ensure that the source and mask are of same shape"
            y_range, x_range = source.shape  
            
        mat_A = self.laplacian_matrix(y_range, x_range)

        laplacian = mat_A.tocsc()

        # set the region outside the mask to identity    
        for x in range(1, y_range - 1):
            for y in range(1, x_range - 1):
                if mask[x, y] == 0:
                    k = y + x * x_range
                    mat_A[k, k] = 1
                    mat_A[k, k + 1] = 0
                    mat_A[k, k - 1] = 0
                    mat_A[k, k + x_range] = 0
                    mat_A[k, k - x_range] = 0

        mat_A = mat_A.tocsc()

        mask_flat = mask.flatten()    
        source_flat = source.flatten()       

        if cloning:
            # inside the mask:   
            target_flat = dest.flatten()
            mat_b = laplacian.dot(source_flat)
            if grad_mix: 
                target_grad = laplacian.dot(target_flat)
                for i in range(len(mat_b)):
                    if abs(target_grad[i]) > abs(mat_b[i]):
                        mat_b[i] = target_grad[i]
            # outside the mask:
            mat_b[mask_flat==0] = target_flat[mask_flat==0]
        else: 
            # inside the mask: 
            mat_b = np.zeros(mask_flat.shape)
            
            # outside the mask:
            mat_b[mask_flat==0] = source_flat[mask_flat==0]
        
        x = spsolve(mat_A, mat_b)
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        return x.astype('uint8')
    
    def colour(self, mask, cloning, grad_mix):
        x = np.zeros(self.source.shape)
        source = self.source
        dest = self.dest
        if cloning:
            for channel in range(self.source.shape[2]):
                print(f"performing poisson editing for channel: {channel}")
                x[:,:,channel] = self.poisson_edit(source[:,:,channel],mask,dest[:,:,channel], cloning, grad_mix=grad_mix)
            return x.astype(int)
        else:
            for channel in range(self.source.shape[2]):
                print(f"performing poisson editing for channel: {channel}")
                x[:,:,channel] = self.poisson_edit(source[:,:,channel],mask, dest, cloning, grad_mix=False)
            return x.astype(int)








