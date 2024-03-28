'''
Test for overlapping components based on their temporal and spatial correlations across z-planes

'''
import tifffile
import os
import matplotlib.pyplot as plt

# load local package
root_dir = os.path.split(os.getcwd())[0] # get root
io_dir = os.path.join(root_dir,'code','io') # get utils folder path
import sys
sys.path.append(io_dir) # add it to system path (not ideal) - doing this to reduce pip installs for local lab usage

# import converter
import raw_to_tif as r2t # import movie

# filepath to .raw file - temporaily storing on desktop to avoid any possibility of corrupting og data on removeable drive
filepath = r"C:\Users\spell\Desktop\John\cleanLines1_img\Image_001_001.raw"
self = r2t.RawToTif(filepath=filepath)

# test suite2p conversion
def test1(self):
    self.convert(method='suite2p')
test1(self)

# test max projection over z-plane conversion
def test2(self):
    self.convert(method='max_proj')
test2(self)

def test3(self):
    self.convert(method='4D')
test3(self)


# load in file from matlab vs python to make sure they checkout
mat = tifffile.memmap(r"C:\Users\spell\Desktop\John\cleanLines1_img\img.tif")
pyt = tifffile.memmap(r"C:\Users\spell\Desktop\John\cleanLines1_img\img_mmap_maxproj_z.tif")

#
matfile = tifffile.memmap(r"D:\Maya\cat2.0\PFC\A02\cleanLines1_img\img.tif")
plt.imshow(matfile[-1])
