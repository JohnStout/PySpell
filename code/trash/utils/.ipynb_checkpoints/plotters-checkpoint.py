import matplotlib.pyplot as plt
import numpy as np
import cv2

# technique to rescale the image easily to maximize viewing capacity
def scaled_imshow(image):
    plt_data = plt.imshow(image, 
                          vmin=np.percentile(np.ravel(image),50), 
                          vmax=np.percentile(np.ravel(image),99.5))
    return plt_data

# gaussian filter over the image
def gauss_filt(image, gSig_filt):
    #gSig_filt = np.array([7,7])
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    ex = cv2.filter2D(np.array(image, dtype=np.float32),-1, ker2D, borderType=cv2.BORDER_REFLECT)
    return ex