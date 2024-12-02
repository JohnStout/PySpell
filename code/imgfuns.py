import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interp_img(img):
    '''
    Translated from Matlab to Python thanks to CoPilot :)

    Args:
        >>> img: Y x X x Z shaped data with Z representing 3 images, the middle one needing interpolation.
                    if your data is of shape ZYX, just run np.moveaxis(array,0,-1)

    Outputs:
        >>> img_interp: interpolated image data (2nd image) based on the first and last image provided
        
    NOTE: if your array differs from the expected 512 x 512 XY pixel dimensions, you may have to fact check that the output is correct!

    '''
    # Extract the first and last slices
    A1 = img[:, :, 0]
    A3 = img[:, :, 2]
    
    # Create a new 3D matrix to store the interpolated data
    A_interp = np.zeros((img.shape[0],img.shape[1], img.shape[2]))
    
    # Assign the first and last slices to the new matrix
    A_interp[:, :, 0] = A1
    A_interp[:, :, 2] = A3
    
    # Create the grid for interpolation
    X, Y = np.arange(img.shape[0]), np.arange(img.shape[1])
    Z = np.array([0, 2])
    Xq, Yq, Zq = np.meshgrid(X, Y, np.arange(3), indexing='ij')
    
    # Perform the interpolation
    points = (X, Y, Z)
    interpolator = RegularGridInterpolator(points, img[:, :, [0, 2]], method='linear')
    img_interp = interpolator((Xq, Yq, Zq))

    # return 16 bit precision
    img_interp = img_interp.astype(np.int16)

    return img_interp