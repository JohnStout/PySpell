import os
import numpy as np
import tifffile as tf

# TODO: this function will perform a spatial correlation analysis of z-plane components
# as well as a temporal correlation to identify components that are likely the same cells
# Maybe build in some threshold that takes the maximum projection of the component or just simply ignores the component in plane X
def test_for_duplicate_zplane_components():
    pass


fname = r'C:\Users\spell\Desktop\John\cleanLines1_img\suite2p'
def suite2p_z_plane(suite2p_path: str): 
    '''
    Args:
        >>> suite2p_path: path to the planes saved out by suite2p
    '''

    # list folder contents
    folder_contents = os.listdir(suite2p_path)

    # get paths with the name 'plane' in the title
    fnames = [os.path.join(suite2p_path,i) for i in folder_contents if 'plane' in i] # get directories with 'plane' in title
    
    # get ops data from each plane of suite2p data
    ops = [np.load(os.path.join(i,'ops.npy'),allow_pickle=True).item() for i in fnames] # load ops data across all directories
    max_proj = [i['max_proj'] for i in ops]
    mean_img = [i['meanImg'] for i in ops]

    # save data as tif
    zname = os.path.join(os.path.split(suite2p_path)[0],'suite2p_z.tif')
    z = len(mean_img); y = mean_img[0].shape[0]; x = mean_img[0].shape[1]
    im = tf.memmap(
        zname,
        shape=(z,y,x),
        dtype=np.uint16,
        imagej=True
        #append=True
    )
    # chunk write
    for z in range(len(mean_img)):
        im[z,:,:]=mean_img[z]
        im.flush()

    