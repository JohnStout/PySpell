'''
Tests for PySpell
'''
import sys
import os

# load local package
root_dir = os.path.split(os.getcwd())[0] # get root
io_dir = os.path.join(root_dir,'code') # get utils folder path
sys.path.append(io_dir) # add it to system path (not ideal) - doing this to reduce pip installs for local lab usage

# import converter
from dataWrangler import matpy

# this method blindly parses your .mat file into a python dictionary
matpath_2p  = r"C:\Users\uggriffin\SpellmanLab Dropbox\timspellman\Imaging\PFCMDTReanalysis\Imaging\t284\SEDS2SEDS3\sourcesS2P.mat"
matpath_beh = r"C:\Users\uggriffin\SpellmanLab Dropbox\timspellman\Imaging\PFCMDTReanalysis\Behavior2P\t284\SEDS2SEDS3Beh.mat"
neuron, vars     = matpy(matpath_2p).parseNeuronStruct()
behdata, behvars = matpy(matpath_beh).parsemat()
