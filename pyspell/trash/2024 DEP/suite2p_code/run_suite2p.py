import suite2p
import tifffile
import xmltodict
import numpy as np
import os

# load suite2p results
# CHANGE ME
movie_path = r"C:\Users\spell\Desktop\John\cleanLines1_img\img_zplane_2.tif"
gcamp = '6f' # change if sensor is different

#___________________________________________#

# movies and associated frame rates
root_path = os.path.split(movie_path)[0]
movie_name = os.path.split(movie_path)[1]

# load movie
images=tifffile.memmap(movie_path)

# get metadata
root_contents = os.listdir(root_path)
metadata_file = [i for i in root_contents if '.xml' in i][0]
metadata_path = os.path.join(root_path,metadata_file)
file = xmltodict.parse(open(metadata_path,"r").read()) # .xml file

# define frame rate based on metadata
fr = float(file['ThorImageExperiment']['LSM']['@frameRate'])

# default ops
ops = suite2p.default_ops()
ops['fs']=fr
# gcamp
if '6f' in gcamp or '8f' in gcamp: # check the 8f
    ops['tau'] = 0.7 # gcampe6f
elif '6m' in gcamp:
    ops['tau'] = 1.0
elif '6s' in gcamp:
    ops['tau'] = 1.3
ops['save_NWB']=False # set to false for now

# if the shape of your images data is > 3, then you have a z-plane
if len(images.shape) > 3 and len(images.shape) < 5:
    print("z-plane detected. If this is not true, stop and troubleshoot")
    ops['nplanes']=images.shape[-1]
ops['nplanes']=3

# set db, this overrides the ops variable
db = {
    'data_path': [root_path],
    'tiff_list': [movie_name],
}
db

# run suite2p algorithm
output_ops = suite2p.run_s2p(ops=ops, db=db)