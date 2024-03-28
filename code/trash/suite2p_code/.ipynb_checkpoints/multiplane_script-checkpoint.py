# import packages
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile
import xmltodict
import suite2p
import matplotlib.pyplot as plt

# define path
output_path = r"C:\Users\spell\Desktop\John\cleanLines1_img\suite2p"

#
# load data
plane_dirs = [os.path.join(output_path,i) for i in os.listdir(output_path) if 'plane' in i]

# get results
stats_file = []; iscell = []; f_cells = []; f_neuropils = []; spks = []; ops = []
for i in plane_dirs:
    temp_ops = np.load(os.path.join(i,'ops.npy'), allow_pickle=True).item()
    temp_ops.keys() == temp_ops.keys()  
    ops.append(temp_ops)
    stats_file.append(np.load(os.path.join(i,'stat.npy'),allow_pickle=True))
    iscell.append(np.load(os.path.join(i,'iscell.npy'),allow_pickle=True)[:, 0].astype(bool))
    f_cells.append(np.load(os.path.join(i,'F.npy'),allow_pickle=True))
    f_neuropils.append(np.load(os.path.join(i,'Fneu.npy'),allow_pickle=True))
    spks.append(np.load(os.path.join(i,'spks.npy'),allow_pickle=True))

# get image masks
im = []
for i in range(len(stats_file)):
    temp = suite2p.ROI.stats_dicts_to_3d_array(stats_file[i], Ly=ops[i]['Ly'], Lx=ops[i]['Lx'], label_id=True)
    temp[temp == 0] = np.nan
    im.append(temp)

# get single matrix per plane
plane_mask = []
for i in range(len(im)):
    plane_mask.append(np.nanmax(im[i][iscell[i]], axis=0))

#
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(8,8))
for i in range(len(plane_mask)):
    ax[i].imshow(plane_mask[i])

fig2, ax2 = plt.subplots(nrows=1,ncols=3,figsize=(8,8))
for i in range(len(plane_mask)):
    ax2[i].imshow(ops[i]['max_proj'])

#
from suite2p.detection import stats
#stats.filter_overlappers(ypixs, xpixs, overlap_image, max_overlap)

# x and y data from 1 cell
x = stats_file[0][iscell[0]][0]['xpix']
y = stats_file[0][iscell[0]][0]['ypix']
overlap = stats_file[0][iscell[0]][0]['overlap']

#
#overlap_image = 

